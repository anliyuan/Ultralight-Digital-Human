"""训练 UNet 用的数据集。

每条样本包含：
- target_T:      当前帧的人脸下半部分（160×160），作为生成目标
- img_concat_T:  6 通道输入 = 随机参考帧（提供身份信息） + 涂黑嘴部的当前帧
- audio_feat:    当前帧前后各 4 帧的音频特征
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from face_utils import (
    count_jpgs,
    extract_inner,
    gather_audio_window,
    hwc_to_chw_tensor,
    load_face_crop,
    mask_mouth,
    reshape_audio_feat,
)


_AUDIO_FEAT_FILE = {
    "wenet": "aud_wenet.npy",
    "hubert": "aud_hu.npy",
}


class MyDataset(Dataset):
    def __init__(self, dataset_dir: str, mode: str):
        if mode not in _AUDIO_FEAT_FILE:
            raise ValueError(f"Unknown asr mode: {mode}")
        self.mode = mode

        full_body_dir = os.path.join(dataset_dir, "full_body_img")
        landmarks_dir = os.path.join(dataset_dir, "landmarks")
        n_frames = count_jpgs(full_body_dir)
        self.img_path_list = [os.path.join(full_body_dir, f"{i}.jpg") for i in range(n_frames)]
        self.lms_path_list = [os.path.join(landmarks_dir, f"{i}.lms") for i in range(n_frames)]

        audio_feats_path = os.path.join(dataset_dir, _AUDIO_FEAT_FILE[mode])
        self.audio_feats = np.load(audio_feats_path).astype(np.float32)

    def __len__(self) -> int:
        # 以音频特征和图像帧数中较小者为准，避免越界
        return min(self.audio_feats.shape[0], len(self.img_path_list))

    def _build_target_and_masked(self, idx: int):
        """加载当前帧，返回 (target_T, masked_T)。"""
        face_crop = load_face_crop(self.img_path_list[idx], self.lms_path_list[idx])
        inner = extract_inner(face_crop)
        target_T = hwc_to_chw_tensor(inner.copy())
        masked_T = hwc_to_chw_tensor(mask_mouth(inner))
        return target_T, masked_T

    def _build_reference(self, idx: int) -> torch.Tensor:
        """随机取另一帧作为参考。"""
        ref_idx = random.randint(0, len(self) - 1)
        face_crop = load_face_crop(self.img_path_list[ref_idx], self.lms_path_list[ref_idx])
        return hwc_to_chw_tensor(extract_inner(face_crop))

    def __getitem__(self, idx: int):
        target_T, masked_T = self._build_target_and_masked(idx)
        ref_T = self._build_reference(idx)
        img_concat_T = torch.cat([ref_T, masked_T], dim=0)

        audio_feat = gather_audio_window(self.audio_feats, idx)
        audio_feat = reshape_audio_feat(audio_feat, self.mode)

        return img_concat_T, target_T, audio_feat
