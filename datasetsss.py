import os
import cv2
import torch
import random
import numpy as np
import random
from pathlib import Path

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

SYNC_SEQ_LEN = 5
SYNC_SEQ_HALF = SYNC_SEQ_LEN // 2


class MyDataset(Dataset):

    def __init__(self, img_dir, mode, use_syncnet=False):

        self.img_path_list = []
        self.lms_path_list = []
        self.frame_ids = []
        self.mode = mode  # wenet or hubert
        self.img_dir = Path(img_dir)
        self.use_syncnet = use_syncnet
        
        img_root = self.img_dir / "full_body_img"
        lms_root = self.img_dir / "landmarks"
        img_files = sorted(img_root.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
        for img_path in img_files:
            if not img_path.stem.isdigit():
                print(f"[WARN] Non-numeric frame name {img_path.name}, skip this sample.")
                continue
            lms_path = lms_root / f"{img_path.stem}.lms"
            if not lms_path.exists():
                print(f"[WARN] Missing landmark file for frame {img_path.name}, skip this sample.")
                continue
            frame_id = int(img_path.stem)
            self.img_path_list.append(str(img_path))
            self.lms_path_list.append(str(lms_path))
            self.frame_ids.append(frame_id)
        
        if self.mode == "wenet":
            self.audio_feats = np.load(self.img_dir / "aud_wenet.npy")
        elif self.mode == "hubert":
            self.audio_feats = np.load(self.img_dir / "aud_hu.npy")
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
            
        self.audio_feats = self.audio_feats.astype(np.float32)
        valid_indices = [i for i, frame_id in enumerate(self.frame_ids) if frame_id < self.audio_feats.shape[0]]
        if len(valid_indices) != len(self.frame_ids):
            print(f"[WARN] Some frames exceed available audio features ({self.audio_feats.shape[0]}). Skipping out-of-range frames.")
        self.img_path_list = [self.img_path_list[i] for i in valid_indices]
        self.lms_path_list = [self.lms_path_list[i] for i in valid_indices]
        self.frame_ids = [self.frame_ids[i] for i in valid_indices]
        self.sample_count = len(self.img_path_list)
        if self.sample_count == 0:
            raise ValueError(f"No valid samples found in dataset: {img_dir}")
        if len(self.img_path_list) != self.audio_feats.shape[0]:
            print(f"[WARN] Valid frame sample count ({len(self.img_path_list)}) and audio feature count ({self.audio_feats.shape[0]}) differ. Using frame-id alignment.")
        
    def __len__(self):
        return self.sample_count

    def _read_landmarks(self, lms_path):
        lms_list = []
        with open(lms_path, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                arr = line.split(" ")
                arr = np.array(arr, dtype=np.float32)
                lms_list.append(arr)
        lms = np.array(lms_list, dtype=np.int32)
        if lms.shape[0] <= 52:
            raise ValueError(f"Invalid landmark count in {lms_path}: got {lms.shape[0]}")
        return lms

    def _crop_face(self, img, lms, source_name):
        if img is None:
            raise ValueError(f"Failed to read image: {source_name}")
        xmin = int(lms[1][0])
        ymin = int(lms[52][1])
        xmax = int(lms[31][0])
        width = xmax - xmin
        if width <= 0:
            raise ValueError(f"Invalid crop box in {source_name}: xmin={xmin}, xmax={xmax}")
        ymax = ymin + width
        h, w = img.shape[:2]
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Crop box out of bounds in {source_name}")
        crop_img = img[ymin:ymax, xmin:xmax]
        if crop_img.size == 0:
            raise ValueError(f"Empty crop in {source_name}")
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
        return crop_img
    
    def get_audio_features(self, features, index):  # 在当前音频帧前后各取4帧音频特征
        left = index - 4
        right = index + 4
        pad_left = 0
        pad_right = 0
        if left < 0:
            pad_left = -left
            left = 0
        if right > features.shape[0]:
            pad_right = right - features.shape[0]
            right = features.shape[0]
        auds = torch.from_numpy(features[left:right])
        if pad_left > 0:
            auds = torch.cat([torch.zeros_like(auds[:pad_left]), auds], dim=0)
        if pad_right > 0:
            auds = torch.cat([auds, torch.zeros_like(auds[:pad_right])], dim=0) # [8, 16]
        return auds
    
    
    def process_img(self, img, lms_path, img_ex, lms_path_ex):
        lms = self._read_landmarks(lms_path)
        crop_img = self._crop_face(img, lms, lms_path)
        # resize后保留边缘的4个像素，如果视频分辨率比较大的话 建议把resize值和这个值都改大 但宽高必须能被16整除，同时模型结构也要改
        img_real = crop_img[4:164, 4:164].copy() # 保留边缘的4个像素防止贴回去的时候比较违和
        img_real_ori = img_real.copy()
        img_masked = cv2.rectangle(img_real,(5,5,150,145),(0,0,0),-1) # 将图片中间区域涂黑
        
        # 取一张随机图像作为参考和要做推理的图像一起输入 ⬇️⬇️⬇️
        lms = self._read_landmarks(lms_path_ex)
        crop_img = self._crop_face(img_ex, lms, lms_path_ex)
        img_real_ex = crop_img[4:164, 4:164].copy()
        
        img_real_ori = img_real_ori.transpose(2,0,1).astype(np.float32)
        img_masked = img_masked.transpose(2,0,1).astype(np.float32)
        img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
        
        img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
        img_real_T = torch.from_numpy(img_real_ori / 255.0)
        img_masked_T = torch.from_numpy(img_masked / 255.0)
        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)

        return img_concat_T, img_real_T

    def _get_fallback_idx(self, idx):
        if self.__len__() == 1:
            return idx
        for _ in range(self.__len__()):
            fallback_idx = random.randint(0, self.__len__() - 1)
            if fallback_idx != idx:
                return fallback_idx
        return idx

    def _reshape_audio(self, audio_feat):
        if self.mode == "wenet":
            return audio_feat.reshape(128, 16, 32)
        if self.mode == "hubert":
            return audio_feat.reshape(16, 32, 32)
        raise ValueError(f"Unsupported mode: {self.mode}")

    def _build_sync_window(self, center_idx, img_ex, lms_path_ex):
        sync_concats = []
        sync_audios = []
        for offset in range(-SYNC_SEQ_HALF, SYNC_SEQ_HALF + 1):
            ci = center_idx + offset
            if ci < 0:
                ci = 0
            elif ci >= self.sample_count:
                ci = self.sample_count - 1
            s_img = cv2.imread(self.img_path_list[ci])
            s_lms = self.lms_path_list[ci]
            s_concat, _ = self.process_img(s_img, s_lms, img_ex, lms_path_ex)
            sync_concats.append(s_concat)

            s_audio = self.get_audio_features(self.audio_feats, self.frame_ids[ci])
            s_audio = self._reshape_audio(s_audio)
            sync_audios.append(s_audio)
        return torch.stack(sync_concats, dim=0), torch.stack(sync_audios, dim=0)

    def __getitem__(self, idx):
        attempts = 0
        current_idx = idx
        while attempts < self.__len__():
            try:
                img = cv2.imread(self.img_path_list[current_idx])
                lms_path = self.lms_path_list[current_idx]

                ex_int = random.randint(0, self.__len__()-1)
                img_ex = cv2.imread(self.img_path_list[ex_int])
                lms_path_ex = self.lms_path_list[ex_int]

                img_concat_T, img_real_T = self.process_img(img, lms_path, img_ex, lms_path_ex)
                audio_feat = self.get_audio_features(self.audio_feats, self.frame_ids[current_idx])
                audio_feat = self._reshape_audio(audio_feat)

                if self.use_syncnet:
                    sync_concats, sync_audios = self._build_sync_window(
                        current_idx, img_ex, lms_path_ex
                    )
                    return img_concat_T, img_real_T, audio_feat, sync_concats, sync_audios

                return img_concat_T, img_real_T, audio_feat
            except Exception as exc:
                print(f"[WARN] Failed to build sample idx={current_idx}: {exc}")
                attempts += 1
                current_idx = self._get_fallback_idx(current_idx)
        raise RuntimeError(f"Failed to build a valid sample after {attempts} attempts.")
    
        
