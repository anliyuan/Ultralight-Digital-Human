import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasetsss import MyDataset
from syncnet import SyncNet_color
from unet import Model
import random
import torchvision.models as models

def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


device = get_best_device()
print(f"[INFO] Training device: {device}")

def get_args():
    parser = argparse.ArgumentParser(description='Train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--use_syncnet', action='store_true', help="if use syncnet, you need to set 'syncnet_checkpoint'")
    parser.add_argument('--syncnet_checkpoint', type=str, default="")
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--save_dir', type=str, help="trained model save path.")
    parser.add_argument('--see_res', action='store_true')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--asr', type=str, default="hubert")
    parser.add_argument('--sync_loss_weight', type=float, default=2.0,
                        help="syncnet loss weight when --use_syncnet is enabled.")

    return parser.parse_args()

args = get_args()
use_syncnet = args.use_syncnet
# Loss functions
class PerceptualLoss():

    def contentFunc(self):
        conv_3_3_layer = 14
        try:
            cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        except Exception as exc:
            print(f"[WARN] Failed to download VGG19 weights ({exc}); "
                  f"falling back to untrained VGG19 for perceptual loss.")
            cnn = models.vgg19(weights=None).features
        cnn = cnn.to(device)
        model = nn.Sequential()
        model = model.to(device)
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        for p in model.parameters():
            p.requires_grad = False
        return model.eval()

    def __init__(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss

def cosine_loss(a, v, y):
    return nn.CosineEmbeddingLoss(margin=0.2)(a, v, y.view(-1))

def train(net, epoch, batch_size, lr):
    content_loss = PerceptualLoss(torch.nn.MSELoss())
    if use_syncnet:
        if args.syncnet_checkpoint == "":
            raise ValueError("Using syncnet, you need to set 'syncnet_checkpoint'.Please check README")
            
        syncnet = SyncNet_color(args.asr).eval().to(device)
        syncnet.load_state_dict(torch.load(args.syncnet_checkpoint, map_location=device))
        for param in syncnet.parameters():
            param.requires_grad = False
    save_dir= args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    dataloader_list = []
    dataset_list = []
    dataset_dir_list = [args.dataset_dir]
    for dataset_dir in dataset_dir_list:
        dataset = MyDataset(dataset_dir, args.asr, use_syncnet=use_syncnet)
        train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=False, num_workers=4)
        dataloader_list.append(train_dataloader)
        dataset_list.append(dataset)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = nn.L1Loss()

    for e in range(epoch):
        net.train()
        random_i = random.randint(0, len(dataset_dir_list)-1)
        dataset = dataset_list[random_i]
        train_dataloader = dataloader_list[random_i]

        with tqdm(total=len(dataset), desc=f'Epoch {e + 1}/{epoch}', unit='img') as p:
            for batch in train_dataloader:
                if use_syncnet:
                    imgs, labels, audio_feat, sync_concats, sync_audios = batch
                    sync_concats = sync_concats.to(device)
                    sync_audios = sync_audios.to(device)
                else:
                    imgs, labels, audio_feat = batch
                imgs = imgs.to(device)
                labels = labels.to(device)
                audio_feat = audio_feat.to(device)
                preds = net(imgs, audio_feat)

                loss_PerceptualLoss = content_loss.get_loss(preds, labels)
                loss_pixel = criterion(preds, labels)

                if use_syncnet:
                    B, S = sync_concats.shape[0], sync_concats.shape[1]
                    flat_concats = sync_concats.reshape(B * S, *sync_concats.shape[2:])
                    flat_audios = sync_audios.reshape(B * S, *sync_audios.shape[2:])
                    flat_sync_preds = net(flat_concats, flat_audios)
                    seq_preds = flat_sync_preds.reshape(
                        B, S * flat_sync_preds.shape[1], flat_sync_preds.shape[2], flat_sync_preds.shape[3]
                    )
                    y = torch.ones([B, 1]).float().to(device)
                    a, v = syncnet(seq_preds, audio_feat)
                    sync_loss = cosine_loss(a, v, y)
                    loss = loss_pixel + loss_PerceptualLoss * 0.01 + args.sync_loss_weight * sync_loss
                else:
                    loss = loss_pixel + loss_PerceptualLoss * 0.01
                p.set_postfix(**{'loss (batch)': loss.item()})
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                p.update(imgs.shape[0])
                
        if e % 5 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, str(e)+'.pth'))
        if args.see_res:
            net.eval()
            sample = dataset.__getitem__(random.randint(0, dataset.__len__()))
            img_concat_T, img_real_T, audio_feat = sample[0], sample[1], sample[2]
            img_concat_T = img_concat_T[None].to(device)
            audio_feat = audio_feat[None].to(device)
            with torch.no_grad():
                pred = net(img_concat_T, audio_feat)[0]
            pred = pred.cpu().numpy().transpose(1,2,0)*255
            pred = np.array(pred, dtype=np.uint8)
            img_real = img_real_T.numpy().transpose(1,2,0)*255
            img_real = np.array(img_real, dtype=np.uint8)
            cv2.imwrite("./train_tmp_img/epoch_"+str(e)+".jpg", pred)
            cv2.imwrite("./train_tmp_img/epoch_"+str(e)+"_real.jpg", img_real)
        
            

if __name__ == '__main__':
    
    
    net = Model(6, args.asr).to(device)
    train(net, args.epochs, args.batchsize, args.lr)
