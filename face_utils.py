"""共用的人脸预处理 / 音频窗口工具。

数据加载、训练、推理、流式推理里都需要做的几件事：
- 从 .lms 文件读取关键点
- 根据关键点算嘴部下方的正方形 bbox
- 把 bbox 区域裁出来、resize 到固定尺寸、再取里层喂给网络
- 在音频特征序列上以当前帧为中心取一个上下文窗口

这些动作之前在多个文件里复制粘贴，集中在这里以保证一致性。
"""

from __future__ import annotations

import os
from typing import Tuple

import cv2
import numpy as np
import torch


# 训练 / 推理时，从原图裁出来的人脸区域统一 resize 到 FACE_CROP_SIZE×FACE_CROP_SIZE，
# 再从中取 FACE_INNER_SIZE×FACE_INNER_SIZE 的内部区域作为网络输入 / 标签；
# 留下来的边给"贴回原图"时做一个缓冲，避免边缘违和。
FACE_CROP_SIZE = 168
FACE_INNER_SIZE = 160
FACE_BORDER = (FACE_CROP_SIZE - FACE_INNER_SIZE) // 2  # = 4

# 涂黑嘴部矩形 (x, y, w, h)
MASK_RECT = (5, 5, 150, 145)

# 训练 / 推理时音频上下文半窗口：取当前帧前后各 AUDIO_HALF_WINDOW 帧
AUDIO_HALF_WINDOW = 4

# UNet 输入的音频特征 reshape 目标形状
_AUDIO_FEAT_SHAPE = {
    "wenet": (128, 16, 32),
    "hubert": (16, 32, 32),
}


Bbox = Tuple[int, int, int, int]


def read_landmarks(lms_path: str) -> np.ndarray:
    """从 .lms 文件读关键点。每行 'x y'，返回 int32 ndarray [N, 2]。"""
    pts = []
    with open(lms_path, "r") as f:
        for line in f.read().splitlines():
            line = line.strip()
            if not line:
                continue
            pts.append(np.fromstring(line, sep=" ", dtype=np.float32))
    return np.array(pts, dtype=np.int32)


def compute_face_bbox(landmarks: np.ndarray) -> Bbox:
    """从关键点算出包含嘴部的正方形 bbox (xmin, ymin, xmax, ymax)。

    选点规则沿用原作者：横向以 #1 / #31 关键点为左右边界，纵向以 #52 为上边界，
    并强制为正方形（边长等于横向宽度）。
    """
    xmin = int(landmarks[1][0])
    ymin = int(landmarks[52][1])
    xmax = int(landmarks[31][0])
    width = xmax - xmin
    ymax = ymin + width
    return xmin, ymin, xmax, ymax


def crop_face(img: np.ndarray, bbox: Bbox) -> np.ndarray:
    """裁切 bbox 区域并 resize 到 FACE_CROP_SIZE。"""
    xmin, ymin, xmax, ymax = bbox
    region = img[ymin:ymax, xmin:xmax]
    return cv2.resize(region, (FACE_CROP_SIZE, FACE_CROP_SIZE), interpolation=cv2.INTER_AREA)


def extract_inner(face_crop: np.ndarray) -> np.ndarray:
    """从 FACE_CROP_SIZE 大小的 crop 中取中心的 FACE_INNER_SIZE 部分。"""
    b = FACE_BORDER
    return face_crop[b:b + FACE_INNER_SIZE, b:b + FACE_INNER_SIZE].copy()


def mask_mouth(img: np.ndarray) -> np.ndarray:
    """把 FACE_INNER_SIZE 大小的人脸图嘴部矩形区域涂黑（原地修改并返回）。"""
    return cv2.rectangle(img, MASK_RECT, (0, 0, 0), -1)


def hwc_to_chw_tensor(img_hwc: np.ndarray) -> torch.Tensor:
    """[H, W, 3] uint8 → [3, H, W] float32 tensor，并归一化到 [0, 1]。"""
    img = img_hwc.transpose(2, 0, 1).astype(np.float32) / 255.0
    return torch.from_numpy(img)


def gather_audio_window(
    features: np.ndarray,
    index: int,
    half_window: int = AUDIO_HALF_WINDOW,
) -> torch.Tensor:
    """取以 index 为中心、半径 half_window 的音频特征窗口，越界处用 0 填充。

    返回 shape [2*half_window, ...] 的 tensor，dtype 跟 features 保持一致。
    """
    left = index - half_window
    right = index + half_window
    pad_left = max(0, -left)
    pad_right = max(0, right - features.shape[0])
    left = max(0, left)
    right = min(features.shape[0], right)
    auds = torch.from_numpy(features[left:right])
    if pad_left > 0:
        auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
    if pad_right > 0:
        auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0)
    return auds


def reshape_audio_feat(audio_feat: torch.Tensor, mode: str) -> torch.Tensor:
    """根据 ASR 模式 reshape 音频特征到 UNet 期望的形状。"""
    if mode not in _AUDIO_FEAT_SHAPE:
        raise ValueError(f"Unknown asr mode: {mode}")
    return audio_feat.reshape(*_AUDIO_FEAT_SHAPE[mode])


def count_jpgs(dir_path: str) -> int:
    return sum(1 for f in os.listdir(dir_path) if f.endswith(".jpg"))


def load_face_crop(img_path: str, lms_path: str) -> np.ndarray:
    """便捷函数：读图 + 读关键点 + 算 bbox + 裁切，返回 FACE_CROP_SIZE 大小的 crop。"""
    img = cv2.imread(img_path)
    bbox = compute_face_bbox(read_landmarks(lms_path))
    return crop_face(img, bbox)
