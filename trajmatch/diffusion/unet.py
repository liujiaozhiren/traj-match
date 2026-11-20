from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from trajmatch.diffusion.blocks import AttnBlock, Downsample, ResnetBlock, Upsample, nonlinearity, timestep_embedding


class _TailBody(nn.Module):
    """Two-branch UNet. x branch length 4, condition branch length 8, fused at length 2."""

    def __init__(self, ch: int = 512, out_ch: int = 2, num_res_blocks: int = 2, dropout: float = 0.1, in_channels: int = 2):
        super().__init__()
        self.ch = ch
        self.temb_ch = ch * 4
        self.attn_resolutions = {4, 2}

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([nn.Linear(ch, self.temb_ch), nn.Linear(self.temb_ch, self.temb_ch)])

        self.conv_in_x = nn.Conv1d(in_channels, ch, 3, 1, 1)
        self.x_block0 = nn.ModuleList([ResnetBlock(ch, ch, False, dropout, self.temb_ch) for _ in range(num_res_blocks)])
        self.x_attn0 = nn.ModuleList([AttnBlock(ch) for _ in range(num_res_blocks)])
        self.x_down = Downsample(ch, True)
        ch2 = ch * 2
        self.x_block1 = nn.ModuleList(
            [ResnetBlock(ch, ch2, False, dropout, self.temb_ch)]
            + [ResnetBlock(ch2, ch2, False, dropout, self.temb_ch) for _ in range(num_res_blocks - 1)]
        )
        self.x_attn1 = nn.ModuleList([AttnBlock(ch2) for _ in range(num_res_blocks)])

        self.conv_in_info = nn.Conv1d(in_channels, ch, 3, 1, 1)
        self.i_block0 = nn.ModuleList([ResnetBlock(ch, ch, False, dropout, self.temb_ch) for _ in range(num_res_blocks)])
        self.i_attn0 = nn.ModuleList([AttnBlock(ch) for _ in range(num_res_blocks)])
        self.i_down0 = Downsample(ch, True)
        self.i_block1 = nn.ModuleList(
            [ResnetBlock(ch, ch2, False, dropout, self.temb_ch)]
            + [ResnetBlock(ch2, ch2, False, dropout, self.temb_ch) for _ in range(num_res_blocks - 1)]
        )
        self.i_attn1 = nn.ModuleList([AttnBlock(ch2) for _ in range(num_res_blocks)])
        self.i_down1 = Downsample(ch2, True)
        self.i_block2 = nn.ModuleList([ResnetBlock(ch2, ch2, False, dropout, self.temb_ch) for _ in range(num_res_blocks)])
        self.i_attn2 = nn.ModuleList([AttnBlock(ch2) for _ in range(num_res_blocks)])

        self.mid_block1 = ResnetBlock(ch2, ch2, False, dropout, self.temb_ch)
        self.mid_attn = AttnBlock(ch2)
        self.mid_block2 = ResnetBlock(ch2, ch2, False, dropout, self.temb_ch)

        self.up_block1 = nn.ModuleList(
            [ResnetBlock(ch2 + ch2, ch2, False, dropout, self.temb_ch), ResnetBlock(ch2 + ch2, ch2, False, dropout, self.temb_ch)]
        )
        self.up_attn1 = nn.ModuleList([AttnBlock(ch2) for _ in range(2)])
        self.up = Upsample(ch2, True)
        self.up_block0 = nn.ModuleList(
            [ResnetBlock(ch2 + ch, ch, False, dropout, self.temb_ch), ResnetBlock(ch + ch, ch, False, dropout, self.temb_ch)]
        )
        self.up_attn0 = nn.ModuleList([AttnBlock(ch) for _ in range(2)])
        self.norm_out = group_norm_out(ch)
        self.conv_out = nn.Conv1d(ch, out_ch, 3, 1, 1)

    def forward(self, x, info, t, extra=None):
        temb = timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        if extra is not None:
            temb = temb + extra

        hs = []
        h = self.conv_in_x(x)
        hs.append(h)
        for blk, attn in zip(self.x_block0, self.x_attn0):
            h = blk(h, temb)
            if h.size(-1) in self.attn_resolutions:
                h = attn(h)
            hs.append(h)
        h = self.x_down(h)
        for blk, attn in zip(self.x_block1, self.x_attn1):
            h = blk(h, temb)
            if h.size(-1) in self.attn_resolutions:
                h = attn(h)
            hs.append(h)
        h_x = h

        h = self.conv_in_info(info)
        for blk, attn in zip(self.i_block0, self.i_attn0):
            h = blk(h, temb)
            if h.size(-1) in self.attn_resolutions:
                h = attn(h)
        h = self.i_down0(h)
        for blk, attn in zip(self.i_block1, self.i_attn1):
            h = blk(h, temb)
            if h.size(-1) in self.attn_resolutions:
                h = attn(h)
        h = self.i_down1(h)
        for blk, attn in zip(self.i_block2, self.i_attn2):
            h = blk(h, temb)
            if h.size(-1) in self.attn_resolutions:
                h = attn(h)

        h = h_x + h
        h = self.mid_block1(h, temb)
        if h.size(-1) in self.attn_resolutions:
            h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        for blk, attn in zip(self.up_block1, self.up_attn1):
            h = _cat_skip(h, hs.pop())
            h = blk(h, temb)
            if h.size(-1) in self.attn_resolutions:
                h = attn(h)
        h = self.up(h)
        for blk, attn in zip(self.up_block0, self.up_attn0):
            h = _cat_skip(h, hs.pop())
            h = blk(h, temb)
            if h.size(-1) in self.attn_resolutions:
                h = attn(h)
        return self.conv_out(nonlinearity(self.norm_out(h)))


def group_norm_out(channels: int):
    from trajmatch.diffusion.blocks import group_norm

    return group_norm(channels)


def _cat_skip(h, ht):
    if ht.size(-1) != h.size(-1):
        diff = ht.size(-1) - h.size(-1)
        if diff > 0:
            h = F.pad(h, (0, diff))
        else:
            ht = F.pad(ht, (0, -diff))
    return torch.cat([h, ht], dim=1)


class TailUNet(nn.Module):
    def __init__(self, ch: int = 512, num_res_blocks: int = 2):
        super().__init__()
        self.unet = _TailBody(ch=ch, num_res_blocks=num_res_blocks)

    def forward(self, xt, info, t, extra=None):
        return self.unet(xt, info, t, extra=extra)
