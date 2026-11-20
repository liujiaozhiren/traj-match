from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trajmatch.config import HYDRA_TAIL, TAIL_LEN
from trajmatch.pairs import heads_xy


class _IndexedPairs(Dataset):
    def __init__(self, pairs):
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return idx, self.pairs[idx]


def _collate(batch):
    idxs, pre, post = [], [], []
    for idx, item in batch:
        hp, ho = heads_xy(item)
        idxs.append(int(idx))
        pre.append(hp.t().contiguous())
        post.append(ho.t().contiguous())
    return torch.tensor(idxs, dtype=torch.long), torch.stack(pre), torch.stack(post)


@dataclass
class TailCache:
    hydra_pre: torch.Tensor
    hydra_post: torch.Tensor

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"hydra_pre": self.hydra_pre, "hydra_post": self.hydra_post}, path)

    @staticmethod
    def load(path: str) -> "TailCache":
        data = torch.load(path, map_location="cpu")
        return TailCache(hydra_pre=data["hydra_pre"].float(), hydra_post=data["hydra_post"].float())


@torch.no_grad()
def build_cache(pairs, pre_model, post_model, hydra_tail: int = HYDRA_TAIL, batch: int = 16, device: str = "cpu") -> TailCache:
    loader = DataLoader(_IndexedPairs(pairs), batch_size=batch, shuffle=False, collate_fn=_collate)
    n = len(pairs)
    out_pre = torch.empty((n, hydra_tail, TAIL_LEN, 2))
    out_post = torch.empty((n, hydra_tail, TAIL_LEN, 2))
    pre_model.to(device).eval()
    post_model.to(device).eval()
    for idxs, head_pre, head_post in tqdm(loader, desc="sample tails"):
        sampled_pre = pre_model.sample(head_pre.to(device), num=hydra_tail).permute(0, 1, 3, 2).cpu()
        sampled_post = post_model.sample(head_post.to(device), num=hydra_tail).permute(0, 1, 3, 2).cpu()
        out_pre[idxs] = sampled_pre
        out_post[idxs] = sampled_post
    return TailCache(hydra_pre=out_pre, hydra_post=out_post)
