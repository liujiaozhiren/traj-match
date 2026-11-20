from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from trajmatch.config import MATCH_DIM, MATCH_HEADS, MATCH_LAYERS


class PairMatcher(nn.Module):
    def __init__(self, dim: int = MATCH_DIM, n_heads: int = MATCH_HEADS, n_layers: int = MATCH_LAYERS):
        super().__init__()
        self.point = nn.Linear(2, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, info_pre, info_post, hydra_pre, hydra_post):
        pre = torch.cat([info_pre, hydra_pre.reshape(info_pre.size(0), -1, 2)], dim=1)
        post = torch.cat([info_post, hydra_post.reshape(info_post.size(0), -1, 2)], dim=1)
        tokens = self.point(torch.cat([pre, post], dim=1))
        cls = self.cls.expand(tokens.size(0), -1, -1)
        hidden = self.enc(torch.cat([cls, tokens], dim=1))[:, 0]
        return self.out(hidden).squeeze(-1)


def pairwise_inputs(info_pre, info_post, hydra_pre, hydra_post):
    """Broadcast query rows against gallery rows. Returns tensors of shape (B1*B2, ...)."""
    b1 = info_pre.size(0)
    b2 = info_post.size(0)
    info_q = info_pre.unsqueeze(1).expand(b1, b2, -1, -1).reshape(b1 * b2, info_pre.size(1), 2)
    info_g = info_post.unsqueeze(0).expand(b1, b2, -1, -1).reshape(b1 * b2, info_post.size(1), 2)
    hydra_q = hydra_pre.unsqueeze(1).expand(b1, b2, -1, -1, -1).reshape(b1 * b2, hydra_pre.size(1), hydra_pre.size(2), 2)
    hydra_g = hydra_post.unsqueeze(0).expand(b1, b2, -1, -1, -1).reshape(b1 * b2, hydra_post.size(1), hydra_post.size(2), 2)
    return info_q, info_g, hydra_q, hydra_g


def retrieval_logits(model: PairMatcher, info_pre, info_post, hydra_pre, hydra_post) -> torch.Tensor:
    b1 = info_pre.size(0)
    b2 = info_post.size(0)
    packed = pairwise_inputs(info_pre, info_post, hydra_pre, hydra_post)
    return model(*packed).view(b1, b2)


def bce_pairs(model: PairMatcher, info_pre, info_post, hydra_pre, hydra_post) -> torch.Tensor:
    b = info_post.size(0)
    if b < 2:
        raise ValueError("batch size must be at least 2")
    shift = int(torch.randint(1, b, (1,), device=info_pre.device).item())
    perm = (torch.arange(b, device=info_pre.device) + shift) % b
    info_q = torch.cat([info_pre, info_pre], dim=0)
    info_g = torch.cat([info_post, info_post[perm]], dim=0)
    hydra_q = torch.cat([hydra_pre, hydra_pre], dim=0)
    hydra_g = torch.cat([hydra_post, hydra_post[perm]], dim=0)
    logits = model(info_q, info_g, hydra_q, hydra_g)
    labels = torch.cat([torch.ones(b, device=logits.device), torch.zeros(b, device=logits.device)])
    return F.binary_cross_entropy_with_logits(logits, labels)
