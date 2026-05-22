"""轻量 U-Net 数字人模型。

基于 InvertedResidual 的 5 层下采样 / 4 层上采样 U-Net，音频特征通过 AudioConv*
分支编码后在 bottleneck 处与图像特征拼接、融合。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# 主干通道数。如果想压缩模型用于更弱的移动设备，可以改成 [16, 32, 64, 128, 256]，
# AudioConvWenet / AudioConvHubert 里的通道数也要同步调整。
_MAIN_CHANNELS = [32, 64, 128, 256, 512]


class InvertedResidual(nn.Module):
    """MobileNetV2 风格的 Inverted Residual block。"""

    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        use_res_connect: bool,
        expand_ratio: int = 6,
    ):
        super().__init__()
        assert stride in (1, 2)
        self.use_res_connect = use_res_connect

        hidden_dim = inp * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_res_connect else out


class DoubleConvDW(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        self.double_conv = nn.Sequential(
            InvertedResidual(in_channels, out_channels, stride=stride,
                             use_res_connect=False, expand_ratio=2),
            InvertedResidual(out_channels, out_channels, stride=1,
                             use_res_connect=True, expand_ratio=2),
        )

    def forward(self, x):
        return self.double_conv(x)


class InConvDw(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.inconv = InvertedResidual(in_channels, out_channels, stride=1,
                                       use_res_connect=False, expand_ratio=2)

    def forward(self, x):
        return self.inconv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = DoubleConvDW(in_channels, out_channels, stride=2)

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConvDW(in_channels, out_channels, stride=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.shape[2] - x1.shape[2]
        diff_x = x2.shape[3] - x1.shape[3]
        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AudioConvWenet(nn.Module):
    """wenet 输入：[B, 128, 16, 32] → [B, 512, H', W']"""

    def __init__(self, ch=_MAIN_CHANNELS):
        super().__init__()
        self.conv1 = InvertedResidual(ch[2], ch[3], stride=1, use_res_connect=False, expand_ratio=2)
        self.conv2 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)

        self.conv3 = nn.Conv2d(ch[3], ch[3], kernel_size=3, padding=1, stride=(1, 2))
        self.bn3 = nn.BatchNorm2d(ch[3])

        self.conv4 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)

        self.conv5 = nn.Conv2d(ch[3], ch[4], kernel_size=3, padding=3, stride=2)
        self.bn5 = nn.BatchNorm2d(ch[4])
        self.relu = nn.ReLU()

        self.conv6 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv7 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = self.conv7(x)
        return x


class AudioConvHubert(nn.Module):
    """hubert 输入：[B, 16, 32, 32] → [B, 512, H', W']"""

    def __init__(self, ch=_MAIN_CHANNELS):
        super().__init__()
        self.conv1 = InvertedResidual(16, ch[1], stride=1, use_res_connect=False, expand_ratio=2)
        self.conv2 = InvertedResidual(ch[1], ch[2], stride=1, use_res_connect=False, expand_ratio=2)

        self.conv3 = nn.Conv2d(ch[2], ch[3], kernel_size=3, padding=1, stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(ch[3])

        self.conv4 = InvertedResidual(ch[3], ch[3], stride=1, use_res_connect=True, expand_ratio=2)

        self.conv5 = nn.Conv2d(ch[3], ch[4], kernel_size=3, padding=3, stride=2)
        self.bn5 = nn.BatchNorm2d(ch[4])
        self.relu = nn.ReLU()

        self.conv6 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)
        self.conv7 = InvertedResidual(ch[4], ch[4], stride=1, use_res_connect=True, expand_ratio=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = self.conv7(x)
        return x


_AUDIO_BRANCH = {
    "wenet": AudioConvWenet,
    "hubert": AudioConvHubert,
}


class Model(nn.Module):
    """音频驱动的轻量数字人 UNet。

    输入：
    - x:          [B, n_channels=6, 160, 160]  (BGR ref + masked current)
    - audio_feat: [B, 128, 16, 32] (wenet) 或 [B, 16, 32, 32] (hubert)
    输出：
    - [B, 3, 160, 160]，sigmoid 归一化后的人脸下半部分。
    """

    def __init__(self, n_channels: int = 6, mode: str = "wenet"):
        super().__init__()
        if mode not in _AUDIO_BRANCH:
            raise ValueError(f"Unknown asr mode: {mode}")

        self.n_channels = n_channels
        ch = _MAIN_CHANNELS

        self.audio_model = _AUDIO_BRANCH[mode]()
        self.fuse_conv = nn.Sequential(
            DoubleConvDW(ch[4] * 2, ch[4], stride=1),
            DoubleConvDW(ch[4], ch[3], stride=1),
        )

        self.inc = InConvDw(n_channels, ch[0])
        self.down1 = Down(ch[0], ch[1])
        self.down2 = Down(ch[1], ch[2])
        self.down3 = Down(ch[2], ch[3])
        self.down4 = Down(ch[3], ch[4])

        self.up1 = Up(ch[4], ch[3] // 2)
        self.up2 = Up(ch[3], ch[2] // 2)
        self.up3 = Up(ch[2], ch[1] // 2)
        self.up4 = Up(ch[1], ch[0])

        self.outc = OutConv(ch[0], 3)

    def forward(self, x: torch.Tensor, audio_feat: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        audio_feat = self.audio_model(audio_feat)
        x5 = torch.cat([x5, audio_feat], dim=1)
        x5 = self.fuse_conv(x5)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return torch.sigmoid(self.outc(x))


if __name__ == "__main__":
    # 简单的形状自检，方便确认改完通道后 forward 还能跑通
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--asr", type=str, default="wenet", choices=["wenet", "hubert"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    net = Model(6, args.asr).eval().to(device)
    img = torch.zeros([1, 6, 160, 160], device=device)
    if args.asr == "wenet":
        audio = torch.zeros([1, 128, 16, 32], device=device)
    else:
        audio = torch.zeros([1, 16, 32, 32], device=device)

    with torch.no_grad():
        out = net(img, audio)
    print(f"mode={args.asr}  output shape: {tuple(out.shape)}")
