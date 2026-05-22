"""UNet 训练入口。

checkpoint 保存 {epoch, model, optimizer} 三件套，支持 --resume 断点续训。
"""

from __future__ import annotations

import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasetsss import MyDataset
from unet import Model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True,
                        help="trained model save path.")
    parser.add_argument("--asr", type=str, default="hubert",
                        choices=["wenet", "hubert"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default="",
                        help="path to checkpoint to resume from")
    parser.add_argument("--see_res", action="store_true",
                        help="dump a sample prediction every epoch")
    parser.add_argument("--see_res_dir", type=str, default="./train_tmp_img")
    return parser.parse_args()


class PerceptualLoss:
    """VGG19 conv3_3 特征上的 MSE 感知损失。VGG 仅作特征提取器，禁用梯度。"""

    CONV_3_3_LAYER = 14

    def __init__(self, criterion: nn.Module, device: torch.device):
        self.criterion = criterion
        self.device = device
        self.content_func = self._build_content_func()

    def _build_content_func(self) -> nn.Sequential:
        cnn = tv_models.vgg19(pretrained=True).features.to(self.device).eval()
        for p in cnn.parameters():
            p.requires_grad_(False)
        model = nn.Sequential()
        for i, layer in enumerate(cnn):
            model.add_module(str(i), layer)
            if i == self.CONV_3_3_LAYER:
                break
        return model.to(self.device).eval()

    def __call__(self, fake_im: torch.Tensor, real_im: torch.Tensor) -> torch.Tensor:
        f_fake = self.content_func(fake_im)
        with torch.no_grad():
            f_real = self.content_func(real_im)
        return self.criterion(f_fake, f_real)


def save_checkpoint(path: str, model: nn.Module, optimizer: optim.Optimizer, epoch: int):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def resume_if_any(
    resume_path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> int:
    """返回起始 epoch。空路径表示从头训练。"""
    if not resume_path:
        return 0
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
    ckpt = torch.load(resume_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
    else:
        model.load_state_dict(ckpt)
        start_epoch = 0
    print(f"[resume] loaded from {resume_path}, start_epoch={start_epoch}")
    return start_epoch


def compute_total_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    pixel_criterion: nn.Module,
    perceptual_loss: PerceptualLoss,
) -> torch.Tensor:
    loss_pixel = pixel_criterion(preds, labels)
    loss_perceptual = perceptual_loss(preds, labels)
    return loss_pixel + 0.01 * loss_perceptual


def train_one_epoch(
    net: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    pixel_criterion: nn.Module,
    perceptual_loss: PerceptualLoss,
    device: torch.device,
    progress_desc: str,
    dataset_len: int,
):
    net.train()
    with tqdm(total=dataset_len, desc=progress_desc, unit="img") as p:
        for imgs, labels, audio_feat in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            audio_feat = audio_feat.to(device)

            preds = net(imgs, audio_feat)
            loss = compute_total_loss(preds, labels, pixel_criterion, perceptual_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            p.set_postfix({"loss (batch)": loss.item()})
            p.update(imgs.shape[0])


def dump_sample(net: nn.Module, dataset, save_dir: str, epoch: int, device: torch.device):
    net.eval()
    idx = random.randint(0, len(dataset) - 1)
    img_concat_T, target_T, audio_feat = dataset[idx]
    img_concat_T = img_concat_T[None].to(device)
    audio_feat = audio_feat[None].to(device)
    with torch.no_grad():
        pred = net(img_concat_T, audio_feat)[0]
    pred_img = (pred.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    real_img = (target_T.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f"epoch_{epoch}.jpg"), pred_img)
    cv2.imwrite(os.path.join(save_dir, f"epoch_{epoch}_real.jpg"), real_img)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.see_res:
        os.makedirs(args.see_res_dir, exist_ok=True)

    dataset = MyDataset(args.dataset_dir, args.asr)
    loader = DataLoader(
        dataset, batch_size=args.batchsize, shuffle=True,
        drop_last=False, num_workers=args.num_workers,
    )

    net = Model(6, args.asr).to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pixel_criterion = nn.L1Loss()
    perceptual_loss = PerceptualLoss(nn.MSELoss(), device=device)

    start_epoch = resume_if_any(args.resume, net, optimizer, device)

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(
            net, loader, optimizer,
            pixel_criterion, perceptual_loss,
            device,
            progress_desc=f"Epoch {epoch + 1}/{args.epochs}",
            dataset_len=len(dataset),
        )

        is_save_epoch = (epoch + 1) % args.save_every == 0
        if is_save_epoch or epoch == args.epochs - 1:
            save_checkpoint(os.path.join(args.save_dir, f"{epoch}.pth"), net, optimizer, epoch)
            save_checkpoint(os.path.join(args.save_dir, "last.pth"), net, optimizer, epoch)

        if args.see_res:
            dump_sample(net, dataset, args.see_res_dir, epoch, device)


if __name__ == "__main__":
    main()
