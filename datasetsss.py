import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict


class MyDataset(Dataset):
    def __init__(self, root_dir, mode="hubert"):
        """
        Initialize the dataset with multiple video folders.
        
        Args:
            root_dir (str): Path to the directory containing multiple video folders.
            mode (str): "wenet" or "hubert".
        """
        self.root_dir = root_dir
        self.mode = mode
        self.video_dict = defaultdict()

        # Traverse all subdirectories (each subdirectory corresponds to a video)
        for video_name in sorted(os.listdir(root_dir)):
            video_dir = os.path.join(root_dir, video_name)
            if not os.path.isdir(video_dir):
                continue

            img_dir = os.path.join(video_dir, "full_body_img")
            lms_dir = os.path.join(video_dir, "landmarks")

            # Check if required directories exist
            if not os.path.exists(img_dir) or not os.path.exists(lms_dir):
                print(f"[WARNING] Skipping invalid video folder: {video_dir}")
                continue

            # Collect image and landmarks paths
            img_paths = sorted(
                [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")],
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
            )
            lms_paths = sorted(
                [os.path.join(lms_dir, f) for f in os.listdir(lms_dir) if f.endswith(".lms")],
                key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
            )

            # Load audio features path
            if mode == "wenet":
                aud_path = os.path.join(video_dir, "aud_wenet.npy")
            elif mode == "hubert":
                aud_path = os.path.join(video_dir, "aud_hu_tiny.npy")
            else:
                raise ValueError("Invalid mode. Must be 'wenet' or 'hubert'.")

            if not os.path.exists(aud_path):
                print(f"[WARNING] Skipping video folder due to missing audio file: {video_dir}")
                continue

            # Ensure consistency between images and audio features
            audio_feats = np.load(aud_path).astype(np.float32)
            if audio_feats.shape[0] < 30 or len(img_paths) < 30:
                print(f"[WARNING] Skipping video folder due to too short audio file: {video_dir}")
                continue

            # frame_len = len(img_paths)

            print(f"{video_name}: img len: {len(img_paths)}, aud len: {audio_feats.shape[0]}, lms len: {len(lms_paths)}")

            assert len(img_paths) == len(lms_paths)
            if abs(len(img_paths) - audio_feats.shape[0]) > 2:
                print(f"{video_name} is not consistent.")
                continue

            # Store paths in the dictionary
            self.video_dict[video_name] = {
                "img_paths": img_paths,
                "lms_paths": lms_paths,
                "audio_feats": aud_path
            }

        # Ensure the dataset is not empty
        if len(self.video_dict) == 0:
            raise ValueError("No valid videos found in the root directory.")

        # Define augmentation transforms
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)
        ])
        self.aug_imgs_flags = True
        self.video_dict_list = list(self.video_dict.keys())

        total_frames_len = 0
        self.cumulative_lengths = []
        for video_name in self.video_dict_list:
            frame_len = len(self.video_dict[video_name]["img_paths"])
            total_frames_len += frame_len
            self.cumulative_lengths.append(total_frames_len)

    def __len__(self):
        # Return the number of videos in the dataset
        return self.cumulative_lengths[-1] - 1

    def get_audio_features(self, features, index):
        # load audio features
        features = np.load(features).astype(np.float32)

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
            auds = torch.cat([torch.zeros((pad_left, auds.shape[1], auds.shape[2])), auds], dim=0)
        if pad_right > 0:
            auds = torch.cat([auds, torch.zeros((pad_right, auds.shape[1], auds.shape[2]))], dim=0)

        return auds

    def process_img(self, img, lms_path, img_ex, lms_path_ex):
        lms_list = []
        with open(lms_path, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                arr = line.split(" ")
                arr = np.array(arr, dtype=np.float32)
                lms_list.append(arr)

        lms = np.array(lms_list, dtype=np.int32)
        xmin = lms[1][0]
        ymin = lms[52][1]

        xmax = lms[31][0]
        width = xmax - xmin
        ymax = ymin + width

        random_pixel_shift_x = random.randint(-5, 5)
        random_pixel_shift_y = random.randint(-5, 5)
        crop_img = img[ymin - width // 2 + random_pixel_shift_y:ymax + random_pixel_shift_y, xmin + random_pixel_shift_x:xmax + random_pixel_shift_x]
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
        img_real = crop_img[4:164, 4:164].copy()
        img_real_ori = img_real.copy()
        img_masked = cv2.rectangle(img_real,(10,50,140,100),(0,0,0),-1)

        lms_list = []
        with open(lms_path_ex, "r") as f:
            lines = f.read().splitlines()
            for line in lines:
                arr = line.split(" ")
                arr = np.array(arr, dtype=np.float32)
                lms_list.append(arr)

        lms = np.array(lms_list, dtype=np.int32)
        xmin = lms[1][0]
        ymin = lms[52][1]
        
        xmax = lms[31][0]
        width = xmax - xmin
        ymax = ymin + width
        
        random_pixel_shift_x = random.randint(-5, 5)
        random_pixel_shift_y = random.randint(-5, 5)

        crop_img = img_ex[ymin - width // 2 + random_pixel_shift_y:ymax + random_pixel_shift_y, xmin + random_pixel_shift_x:xmax + random_pixel_shift_x]
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
        img_real_ex = crop_img[4:164, 4:164].copy()

        img_real_ori = img_real_ori.transpose(2,0,1).astype(np.float32)
        img_masked = img_masked.transpose(2,0,1).astype(np.float32)
        # cv2.imwrite("./masked.jpg", img_masked.transpose(1,2,0))
        img_real_ex = img_real_ex.transpose(2,0,1).astype(np.float32)
        
        img_real_ex_T = torch.from_numpy(img_real_ex / 255.0)
        img_real_T = torch.from_numpy(img_real_ori / 255.0)
        img_masked_T = torch.from_numpy(img_masked / 255.0)

        if self.aug_imgs_flags:
            bs_ = img_real_ex_T.size(0)
            bs_2 = img_masked_T.size(0)
            new_cat_all = torch.stack([img_real_ex_T, img_masked_T, img_real_T], dim=0)
            # print('new shape: ', new_cat_all.size())
            new_cat_all = self.aug_trans(new_cat_all)
            img_real_ex_T = new_cat_all[0]
            img_masked_T = new_cat_all[1]
            img_real_T = new_cat_all[2]

        img_concat_T = torch.cat([img_real_ex_T, img_masked_T], axis=0)

        return img_concat_T, img_real_T

    def get_video_and_index(self, idx):
        # Find which video and frame index the given global index belongs to
        for i, cumulative_length in enumerate(self.cumulative_lengths):
            if idx < cumulative_length:
                video_name = self.video_dict_list[i]
                video_data = self.video_dict[video_name]
                local_idx = idx - (self.cumulative_lengths[i - 1] if i > 0 else 0)
                ref_local_idx = random.randint(0, len(video_data["img_paths"]) - 1)

                return video_name, video_data, local_idx, ref_local_idx

        raise IndexError("Index out of range.")

    def __getitem__(self, index):
        # Randomly select a video and frame index
        video_name, video_data, local_idx, ref_local_idx = self.get_video_and_index(index)
        # Load image, landmarks, and audio features
        img = cv2.imread(video_data["img_paths"][local_idx])
        lms_path = video_data["lms_paths"][local_idx]
        audio_feat = self.get_audio_features(video_data["audio_feats"], local_idx)

        # Randomly select another index for augmentation
        ex_int = ref_local_idx
        img_ex = cv2.imread(video_data["img_paths"][ex_int])
        lms_path_ex = video_data["lms_paths"][ex_int]

        # Process images
        img_concat_T, img_real_T = self.process_img(img, lms_path, img_ex, lms_path_ex)

        # Reshape audio features based on mode
        if self.mode == "wenet":
            audio_feat = audio_feat.reshape(256, 16, 32)
        elif self.mode == "hubert":
            audio_feat = audio_feat.reshape(16, 24, 32)

        return img_concat_T, img_real_T, audio_feat


# 使用示例
if __name__ == "__main__":
    root_dir = "/path/to/video_folders"
    mode = "hubert"  # or "wenet"

    dataset = MultiVideoDataset(root_dir, mode)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

    for img_concat_T, img_real_T, audio_feat in dataloader:
        # 在这里进行模型训练
        pass
