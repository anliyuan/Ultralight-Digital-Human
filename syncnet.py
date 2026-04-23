import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import cv2
import os
import numpy as np
from torch import optim
import random
import argparse
from pathlib import Path

SYNC_SEQ_LEN = 5
SYNC_SEQ_HALF = SYNC_SEQ_LEN // 2
SYNC_NEG_FRAME_MARGIN = 15


def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



class Dataset(object):
    def __init__(self, dataset_dir, mode):
        
        self.img_path_list = []
        self.lms_path_list = []
        self.frame_ids = []
        self.dataset_dir = Path(dataset_dir)
        
        img_root = self.dataset_dir / "full_body_img"
        lms_root = self.dataset_dir / "landmarks"
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
                
        if mode=="wenet":
            audio_feats_path = self.dataset_dir / "aud_wenet.npy"
        elif mode=="hubert":
            audio_feats_path = self.dataset_dir / "aud_hu.npy"
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.audio_feats = np.load(audio_feats_path)
        self.audio_feats = self.audio_feats.astype(np.float32)
        valid_indices = [i for i, frame_id in enumerate(self.frame_ids) if frame_id < self.audio_feats.shape[0]]
        if len(valid_indices) != len(self.frame_ids):
            print(f"[WARN] Some frames exceed available audio features ({self.audio_feats.shape[0]}). Skipping out-of-range frames.")
        self.img_path_list = [self.img_path_list[i] for i in valid_indices]
        self.lms_path_list = [self.lms_path_list[i] for i in valid_indices]
        self.frame_ids = [self.frame_ids[i] for i in valid_indices]
        self.sample_count = len(self.img_path_list)
        if self.sample_count == 0:
            raise ValueError(f"No valid samples found in dataset: {dataset_dir}")
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
    
    def process_img(self, img, lms_path):
        lms = self._read_landmarks(lms_path)
        crop_img = self._crop_face(img, lms, lms_path)
        img_real = crop_img[4:164, 4:164].copy()
        img_real_ori = img_real.copy()
        img_real_ori = img_real_ori.transpose(2,0,1).astype(np.float32)
        img_real_T = torch.from_numpy(img_real_ori / 255.0)
        
        return img_real_T

    def _get_fallback_idx(self, idx):
        if self.__len__() == 1:
            return idx
        for _ in range(self.__len__()):
            fallback_idx = random.randint(0, self.__len__() - 1)
            if fallback_idx != idx:
                return fallback_idx
        return idx

    def _sample_mismatched_index(self, idx):
        if self.__len__() == 1:
            return idx
        anchor_frame_id = self.frame_ids[idx]
        for _ in range(self.__len__() * 2):
            candidate_idx = random.randint(0, self.__len__() - 1)
            if abs(self.frame_ids[candidate_idx] - anchor_frame_id) > SYNC_NEG_FRAME_MARGIN:
                return candidate_idx
        return self._get_fallback_idx(idx)

    def _build_face_window(self, center_idx):
        indices = []
        for offset in range(-SYNC_SEQ_HALF, SYNC_SEQ_HALF + 1):
            ci = center_idx + offset
            if ci < 0:
                ci = 0
            elif ci >= self.sample_count:
                ci = self.sample_count - 1
            indices.append(ci)
        face_list = []
        for ci in indices:
            img = cv2.imread(self.img_path_list[ci])
            lms_path = self.lms_path_list[ci]
            face_T = self.process_img(img, lms_path)
            face_list.append(face_T)
        return torch.cat(face_list, dim=0)

    def build_sample(self, sample_idx, is_positive):
        face_stack = self._build_face_window(sample_idx)
        frame_id = self.frame_ids[sample_idx]
        if is_positive or self.__len__() == 1:
            audio_index = frame_id
            y = torch.tensor([1.0], dtype=torch.float32)
        else:
            mismatch_idx = self._sample_mismatched_index(sample_idx)
            audio_index = self.frame_ids[mismatch_idx]
            y = torch.tensor([-1.0], dtype=torch.float32)

        audio_feat = self.get_audio_features(self.audio_feats, audio_index)
        if self.mode == "wenet":
            audio_feat = audio_feat.reshape(128, 16, 32)
        if self.mode == "hubert":
            audio_feat = audio_feat.reshape(16, 32, 32)
        return face_stack, audio_feat, y

    def __getitem__(self, idx):
        attempts = 0
        current_idx = idx
        while attempts < self.__len__():
            try:
                is_positive = random.random() < 0.5 or self.__len__() == 1
                return self.build_sample(current_idx, is_positive)
            except Exception as exc:
                print(f"[WARN] Failed to build syncnet sample idx={current_idx}: {exc}")
                attempts += 1
                current_idx = self._get_fallback_idx(current_idx)
        raise RuntimeError(f"Failed to build a valid syncnet sample after {attempts} attempts.")


class EvalDataset(Dataset):
    def __init__(self, dataset_dir, mode, max_eval_samples=0):
        super().__init__(dataset_dir, mode)
        if max_eval_samples and max_eval_samples > 0:
            self.eval_pair_count = min(self.sample_count * 2, max_eval_samples)
        else:
            self.eval_pair_count = self.sample_count * 2

    def __len__(self):
        return self.eval_pair_count

    def __getitem__(self, idx):
        sample_idx = idx // 2
        is_positive = idx % 2 == 0
        attempts = 0
        current_idx = sample_idx
        while attempts < self.sample_count:
            try:
                return self.build_sample(current_idx, is_positive)
            except Exception as exc:
                print(f"[WARN] Failed to build eval sample idx={current_idx}: {exc}")
                attempts += 1
                current_idx = self._get_fallback_idx(current_idx)
        raise RuntimeError(f"Failed to build a valid eval sample after {attempts} attempts.")

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonorm_Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv2d(cin, cout, kernel_size, stride, padding),
                            )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class Conv2dTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                            nn.BatchNorm2d(cout)
                            )
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.conv_block(x)
        return self.act(out)

class SyncNet_color(nn.Module):
    def __init__(self, mode):
        super(SyncNet_color, self).__init__()

        face_in_ch = 3 * SYNC_SEQ_LEN
        self.face_encoder = nn.Sequential(
            Conv2d(face_in_ch, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=2, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)
        
        p1 = 128
        p2 = (1, 2)
        if mode == "hubert":
            p1 = 16
            p2 = (2, 2)
        
        self.audio_encoder = nn.Sequential(
            Conv2d(p1, 256, kernel_size=3, stride=1, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            
            Conv2d(256, 256, kernel_size=3, stride=p2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 256, kernel_size=3, stride=2, padding=2),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, face_sequences, audio_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)
        
        return audio_embedding, face_embedding

sync_criterion = nn.CosineEmbeddingLoss(margin=0.2)
def cosine_loss(a, v, y):
    return sync_criterion(a, v, y.view(-1))
    
def train(save_dir, dataset_dir, mode):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    train_dataset = Dataset(dataset_dir, mode=mode)
    train_data_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True,
        num_workers=4)
    device = get_best_device()
    model = SyncNet_color(mode).to(device)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=0.001)
    for epoch in range(40):
        for batch in train_data_loader:
            imgT, audioT, y = batch
            imgT = imgT.to(device)
            audioT = audioT.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            audio_embedding, face_embedding = model(imgT, audioT)
            loss = cosine_loss(audio_embedding, face_embedding, y)
            loss.backward()
            optimizer.step()
        print(epoch, loss.item())
        torch.save(model.state_dict(), os.path.join(save_dir, str(epoch)+'.pth'))


@torch.no_grad()
def evaluate(checkpoint_path, dataset_dir, mode, batch_size=32, max_eval_samples=0):
    device = get_best_device()
    dataset = EvalDataset(dataset_dir, mode=mode, max_eval_samples=max_eval_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = SyncNet_color(mode).to(device).eval()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    pos_scores = []
    neg_scores = []
    correct = 0
    total = 0

    for imgT, audioT, y in dataloader:
        imgT = imgT.to(device)
        audioT = audioT.to(device)
        y = y.to(device)
        audio_embedding, face_embedding = model(imgT, audioT)
        scores = F.cosine_similarity(audio_embedding, face_embedding)
        preds = torch.where(scores >= 0, 1.0, -1.0)
        correct += (preds == y.view(-1)).sum().item()
        total += y.numel()

        pos_mask = y.view(-1) > 0
        neg_mask = ~pos_mask
        if pos_mask.any():
            pos_scores.extend(scores[pos_mask].detach().cpu().tolist())
        if neg_mask.any():
            neg_scores.extend(scores[neg_mask].detach().cpu().tolist())

    pos_mean = float(np.mean(pos_scores)) if pos_scores else float("nan")
    neg_mean = float(np.mean(neg_scores)) if neg_scores else float("nan")
    margin = pos_mean - neg_mean if pos_scores and neg_scores else float("nan")
    accuracy = correct / total if total else 0.0

    print(f"eval_samples={total}")
    print(f"pos_mean={pos_mean:.4f}")
    print(f"neg_mean={neg_mean:.4f}")
    print(f"margin={margin:.4f}")
    print(f"acc_at_zero={accuracy:.4f}")
    return {
        "eval_samples": total,
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "margin": margin,
        "acc_at_zero": accuracy,
    }
            
            
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--asr', type=str)
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--max_eval_samples', type=int, default=0)
    opt = parser.parse_args()
    
    # syncnet = SyncNet_color(mode=opt.asr)
    # img = torch.zeros([1,3,160,160])
    # # audio = torch.zeros([1,128,16,32])
    # audio = torch.zeros([1,16,32,32])
    # audio_embedding, face_embedding = syncnet(img, audio)
    # print(audio_embedding.shape, face_embedding.shape)
    if opt.eval_only:
        if opt.checkpoint == "":
            raise ValueError("Please set --checkpoint when using --eval_only")
        evaluate(opt.checkpoint, opt.dataset_dir, opt.asr, max_eval_samples=opt.max_eval_samples)
    else:
        train(opt.save_dir, opt.dataset_dir, opt.asr)
