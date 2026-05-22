"""离线推理：给定一段音频特征和数据集目录，逐帧合成数字人视频。"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from typing import Optional

import cv2
import numpy as np
import torch

from face_utils import (
    FACE_BORDER,
    FACE_CROP_SIZE,
    FACE_INNER_SIZE,
    compute_face_bbox,
    crop_face,
    extract_inner,
    gather_audio_window,
    hwc_to_chw_tensor,
    mask_mouth,
    read_landmarks,
    reshape_audio_feat,
)
from unet import Model


FPS_BY_MODE = {"hubert": 25, "wenet": 20}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--asr', type=str, default="hubert", choices=["wenet", "hubert"])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--audio_feat', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True,
                        help="output video path (.mp4 / .avi)")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--audio_wav', type=str, default="",
                        help="if provided, ffmpeg will merge audio into the final video")
    return parser.parse_args()


def select_fourcc(save_path: str) -> int:
    """根据输出文件扩展名选择合适的视频编码器。"""
    ext = os.path.splitext(save_path)[1].lower()
    if ext == ".avi":
        return cv2.VideoWriter_fourcc("M", "J", "P", "G")
    return cv2.VideoWriter_fourcc(*"mp4v")  # 默认 mp4v，兼容性比 MJPG 好


def load_model(checkpoint_path: str, mode: str, device: torch.device) -> Model:
    net = Model(6, mode).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    net.load_state_dict(state_dict)
    net.eval()
    return net


def merge_audio(video_path: str, audio_path: str, output_path: str) -> bool:
    if not shutil.which("ffmpeg"):
        print("[warn] ffmpeg not found in PATH, skip audio merging.")
        return False
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    print("[info] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return True


class _FramePicker:
    """按照 0,1,2,...,N-1,N-2,...,1,0,1,... 的来回顺序无限取帧。"""

    def __init__(self, n_frames: int):
        assert n_frames >= 2, "需要至少 2 帧才能来回播放"
        self.n_frames = n_frames
        self.idx = 0
        self.step = 0  # 第一次 next() 时进入 step=1

    def next(self) -> int:
        if self.idx >= self.n_frames - 1:
            self.step = -1
        if self.idx <= 0:
            self.step = 1
        self.idx += self.step
        return self.idx


def _prepare_unet_input(
    img: np.ndarray, lms_path: str, device: torch.device,
):
    """返回 (img_concat_T, crop_img_ori, bbox, original_size)。"""
    bbox = compute_face_bbox(read_landmarks(lms_path))
    crop_h, crop_w = img[bbox[1]:bbox[3], bbox[0]:bbox[2]].shape[:2]

    face_crop = crop_face(img, bbox)
    face_crop_ori = face_crop.copy()
    inner = extract_inner(face_crop)

    # 推理时 reference 用当前帧自己（与训练存在轻微 mismatch，但作者原始设计如此）
    ref_T = hwc_to_chw_tensor(inner.copy()).to(device)
    masked_T = hwc_to_chw_tensor(mask_mouth(inner)).to(device)
    img_concat_T = torch.cat([ref_T, masked_T], dim=0)[None]

    return img_concat_T, face_crop_ori, bbox, (crop_w, crop_h)


def _paste_back(
    img: np.ndarray,
    pred_inner: np.ndarray,
    face_crop_ori: np.ndarray,
    bbox: tuple,
    original_size: tuple,
):
    """把网络预测的 inner 区域贴回原 face crop，再 resize 回原 bbox 大小，覆写 img。"""
    face_crop_ori[FACE_BORDER:FACE_BORDER + FACE_INNER_SIZE,
                  FACE_BORDER:FACE_BORDER + FACE_INNER_SIZE] = pred_inner
    crop_w, crop_h = original_size
    face_crop_ori = cv2.resize(face_crop_ori, (crop_w, crop_h))
    xmin, ymin, xmax, ymax = bbox
    img[ymin:ymax, xmin:xmax] = face_crop_ori


def run(args, device: Optional[torch.device] = None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio_feats = np.load(args.audio_feat)
    img_dir = os.path.join(args.dataset, "full_body_img")
    lms_dir = os.path.join(args.dataset, "landmarks")
    n_imgs = sum(1 for f in os.listdir(img_dir) if f.endswith(".jpg"))
    exm_img = cv2.imread(os.path.join(img_dir, "0.jpg"))
    h, w = exm_img.shape[:2]

    fps = FPS_BY_MODE[args.asr]
    writer = cv2.VideoWriter(args.save_path, select_fourcc(args.save_path), fps, (w, h))

    net = load_model(args.checkpoint, args.asr, device)
    picker = _FramePicker(n_imgs)

    with torch.no_grad():
        for i in range(audio_feats.shape[0]):
            img_idx = picker.next()
            img = cv2.imread(os.path.join(img_dir, f"{img_idx}.jpg"))
            lms_path = os.path.join(lms_dir, f"{img_idx}.lms")

            img_concat_T, face_crop_ori, bbox, original_size = _prepare_unet_input(
                img, lms_path, device,
            )

            audio_feat = gather_audio_window(audio_feats, i)
            audio_feat = reshape_audio_feat(audio_feat, args.asr)[None].to(device)

            pred = net(img_concat_T, audio_feat)[0]
            pred = (pred.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

            _paste_back(img, pred, face_crop_ori, bbox, original_size)
            writer.write(img)
    writer.release()

    if args.audio_wav:
        base, ext = os.path.splitext(args.save_path)
        merged_path = base + "_with_audio" + (ext if ext else ".mp4")
        merge_audio(args.save_path, args.audio_wav, merged_path)
        print(f"[done] video with audio saved to {merged_path}")
    else:
        print(f"[done] video (no audio) saved to {args.save_path}")


if __name__ == "__main__":
    run(parse_args())
