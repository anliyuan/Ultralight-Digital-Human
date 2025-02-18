import os
import cv2
import torch
import random
import numpy as np
import random

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms


class MyDataset(Dataset):
    
    def __init__(self, img_dir, mode):
    
        self.img_path_list = []
        self.lms_path_list = []
        self.mode = mode
        
        for i in range(len(os.listdir(img_dir+"/full_body_img/"))):
            img_path = os.path.join(img_dir+"/full_body_img/", str(i)+".jpg")
            lms_path = os.path.join(img_dir+"/landmarks/", str(i)+".lms")
            self.img_path_list.append(img_path)
            self.lms_path_list.append(lms_path)

        if self.mode == "wenet":
            self.audio_feats = np.load(img_dir+"/aud_wenet.npy")
        if self.mode == "hubert":
            self.audio_feats = np.load(img_dir+"/aud_hu.npy")
            
        self.audio_feats = self.audio_feats.astype(np.float32)
        print(img_dir)
        print(self.audio_feats.shape)
        print(len(self.img_path_list))
        
        if len(self.img_path_list) > self.audio_feats.shape[0]:
            print('audio features and images not match!')
            self.img_path_list = self.img_path_list[:self.audio_feats.shape[0]]
        elif len(self.img_path_list) < self.audio_feats.shape[0]:
            print('audio features and images not match!')
            self.audio_feats = self.audio_feats[-len(self.img_path_list):]

        self.aug_trans = transforms.Compose([
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.1, hue=0.05)
             transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)
        ])
        self.aug_imgs_flags = True


    def __len__(self):
        # return len(self.img_path_list)-1
        # return len(self.img_path_list)
        return self.audio_feats.shape[0]-1
    
    def get_audio_features(self, features, index):
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
    
    def get_audio_features_1(self, features, index):
    
        left = index - 8
        pad_left = 0
        if left < 0:
            pad_left = -left
            left = 0
        auds = features[left:index]
        auds = torch.from_numpy(auds)
        if pad_left > 0:
            # pad may be longer than auds, so do not use zeros_like
            auds = torch.cat([torch.zeros(pad_left, *auds.shape[1:], device=auds.device, dtype=auds.dtype), auds], dim=0)
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

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path_list[idx])
        lms_path = self.lms_path_list[idx]
        
        ex_int = random.randint(0, self.__len__()-1)
        img_ex = cv2.imread(self.img_path_list[ex_int])
        lms_path_ex = self.lms_path_list[ex_int]
        
        img_concat_T, img_real_T = self.process_img(img, lms_path, img_ex, lms_path_ex)
        audio_feat = self.get_audio_features(self.audio_feats, idx) 

        if self.mode == "wenet":
            audio_feat = audio_feat.reshape(256,16,32)
        if self.mode == "hubert":
            audio_feat = audio_feat.reshape(16,32,32)
        
        return img_concat_T, img_real_T, audio_feat
