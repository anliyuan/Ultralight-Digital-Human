import os
import numpy as np
import random
import argparse
import cv2
import torch

from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from models.syncnet import SyncNet_color

class Dataset(object):
    def __init__(self, dataset_dir, mode):
        
        self.img_path_list = []
        self.lms_path_list = []
        
        full_body_img_dir = os.path.join(dataset_dir, "full_body_img")
        landmarks_dir = os.path.join(dataset_dir, "landmarks")
        
        for i in range(len(os.listdir(full_body_img_dir))):

            img_path = os.path.join(full_body_img_dir, str(i)+".jpg")
            lms_path = os.path.join(landmarks_dir, str(i)+".lms")
            self.img_path_list.append(img_path)
            self.lms_path_list.append(lms_path)
                
        if mode=="wenet":
            audio_feats_path = os.path.join(dataset_dir, "aud_wenet.npy")
        if mode=="hubert":
            audio_feats_path = os.path.join(dataset_dir, "aud_hu.npy")
        self.mode = mode
        self.audio_feats = np.load(audio_feats_path)
        self.audio_feats = self.audio_feats.astype(np.float32)
        
    def __len__(self):

        return self.audio_feats.shape[0]-1

    def get_audio_features(self, features, index):
        
        left = index - 8
        right = index + 8
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
        
        crop_img = img[ymin:ymax, xmin:xmax]
        crop_img = cv2.resize(crop_img, (168, 168), cv2.INTER_AREA)
        img_real = crop_img[4:164, 4:164].copy()
        img_real_ori = img_real.copy()
        img_real_ori = img_real_ori.transpose(2,0,1).astype(np.float32)
        img_real_T = torch.from_numpy(img_real_ori / 255.0)
        
        return img_real_T

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path_list[idx])
        lms_path = self.lms_path_list[idx]
        
        ex_int = random.randint(0, self.__len__()-1)
        img_ex = cv2.imread(self.img_path_list[ex_int])
        lms_path_ex = self.lms_path_list[ex_int]
        
        img_real_T = self.process_img(img, lms_path, img_ex, lms_path_ex)
        audio_feat = self.get_audio_features(self.audio_feats, idx) # 
        if self.mode == "wenet":
            audio_feat = audio_feat.reshape(256, 16, 32)
        if self.mode == "hubert":
            audio_feat = audio_feat.reshape(32, 32, 32)

        y = torch.ones(1).float()
        
        return img_real_T, audio_feat, y

logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

def train(save_dir, dataset_dir, epochs, batchsize, num_workers, lr, mode):
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = Dataset(dataset_dir, mode=mode)
    train_data_loader = DataLoader(
        train_dataset, batch_size=batchsize, shuffle=True,
        num_workers=num_workers)
    model = SyncNet_color(mode).cuda()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr)
    for epoch in range(epochs):
        for batch in train_data_loader:
            imgT, audioT, y = batch
            imgT = imgT.cuda()
            audioT = audioT.cuda()
            y = y.cuda()
            audio_embedding, face_embedding = model(imgT, audioT)
            loss = cosine_loss(audio_embedding, face_embedding, y)
            loss.backward()
            optimizer.step()
        print(epoch, loss.item())
        torch.save(model.state_dict(), os.path.join(save_dir, "epoch_" + str(epoch) + '.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="checkpoint/syncnet_ckpt/")
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--asr', type=str, default="hubert")
    opt = parser.parse_args()

    # syncnet = SyncNet_color(mode=opt.asr)
    # img = torch.zeros([1,3,160,160])
    # # audio = torch.zeros([1,128,16,32])
    # audio = torch.zeros([1,16,32,32])
    # audio_embedding, face_embedding = syncnet(img, audio)
    # print(audio_embedding.shape, face_embedding.shape)
    train(opt.save_dir, opt.dataset_dir, opt.epochs, opt.batchsize, opt.num_workers, opt.lr, opt.asr)