from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x):
    return x * torch.sigmoid(x)


def group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


def timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    half = embedding_dim // 2
    freq = torch.exp(torch.arange(half, device=timesteps.device, dtype=torch.float32) * -(np.log(10000) / (half - 1)))
    emb = timesteps.float()[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class Upsample(nn.Module):
    def __init__(self, channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv1d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            x = F.pad(x, (1, 1), mode="constant", value=0)
            return self.conv(x)
        return F.avg_pool1d(x, kernel_size=2, stride=2)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.1, temb_channels=512):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, 1, 1)
        self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, 1, 1)
        if in_channels != out_channels:
            if conv_shortcut:
                self.conv_shortcut = nn.Conv1d(in_channels, out_channels, 3, 1, 1)
            else:
                self.nin_shortcut = nn.Conv1d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, temb):
        h = self.conv1(nonlinearity(self.norm1(x)))
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None]
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = group_norm(channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        b, c, w = q.shape
        attn = torch.bmm(q.permute(0, 2, 1), k) * (int(c) ** -0.5)
        attn = torch.softmax(attn, dim=2).permute(0, 2, 1)
        h = torch.bmm(v, attn).reshape(b, c, w)
        return x + self.proj_out(h)
