from __future__ import annotations

import pickle
import random
from typing import Sequence

import numpy as np
import torch

from trajmatch.config import HEAD_LEN, TAIL_LEN
from trajmatch.coords import points_to_xy


def load_pairs(path: str, max_pairs: int | None = None) -> list:
    raw = pickle.load(open(path, "rb"))
    if not isinstance(raw, list):
        raise TypeError(f"pairs file must be a list, got {type(raw).__name__}")
    if raw and (not isinstance(raw[0], (tuple, list)) or len(raw[0]) != 7):
        raise TypeError("each item must be (mid, A, B, L_A, L_B, F_A, F_B)")
    if max_pairs is not None:
        raw = raw[: int(max_pairs)]
    return list(raw)


def split_pairs(pairs: Sequence, valid_ratio: float, seed: int):
    pairs = list(pairs)
    rng = random.Random(int(seed))
    n = len(pairs)
    k = max(1, int(n * float(valid_ratio)))
    if k >= n:
        k = max(1, n - 1)
    valid_idx = rng.sample(range(n), k)
    valid_set = set(valid_idx)
    train_idx = [i for i in range(n) if i not in valid_set]
    train = [pairs[i] for i in train_idx]
    valid = [pairs[i] for i in valid_idx]
    return train, valid, train_idx, valid_idx


def _window(pts, n: int, reverse: bool) -> list[tuple[float, float]]:
    seq = list(reversed(pts)) if reverse else list(pts)
    return points_to_xy(seq)[:n]


def heads_xy(item, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Observed heads in meters, shape (HEAD_LEN, 2). alpha=1 keeps the noisy head."""
    _, A, B, L_A, L_B, _, _ = item
    pre = np.asarray(_window(A, HEAD_LEN, reverse=False), dtype=np.float32)
    post = np.asarray(_window(B, HEAD_LEN, reverse=True), dtype=np.float32)
    if alpha < 1.0 - 1e-12:
        n_b = len(B)
        label_pre = np.asarray(points_to_xy(L_A[: len(A)])[:HEAD_LEN], dtype=np.float32)
        label_post = np.asarray(
            points_to_xy(list(reversed(L_B[len(L_B) - n_b :])))[:HEAD_LEN], dtype=np.float32
        )
        a = float(alpha)
        pre = a * pre + (1.0 - a) * label_pre
        post = a * post + (1.0 - a) * label_post
    return torch.from_numpy(pre), torch.from_numpy(post)


def diffusion_example(item) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Channel-first head and tail label, each (2, L). Post side is time-reversed."""
    _, A, B, L_A, L_B, _, _ = item
    info_pre = torch.tensor(_window(A, HEAD_LEN, reverse=False), dtype=torch.float32).t().contiguous()
    lab_pre = torch.tensor(points_to_xy(L_A)[-TAIL_LEN:], dtype=torch.float32).t().contiguous()
    info_post = torch.tensor(_window(B, HEAD_LEN, reverse=True), dtype=torch.float32).t().contiguous()
    lab_post = torch.tensor(points_to_xy(list(reversed(L_B)))[-TAIL_LEN:], dtype=torch.float32).t().contiguous()
    return info_pre, lab_pre, info_post, lab_post
