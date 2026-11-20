from __future__ import annotations

import torch


def hr_at(scores: torch.Tensor, k: int) -> float:
    b, n = scores.shape
    if b == 0 or n == 0:
        return 0.0
    kk = min(int(k), n)
    topk = torch.topk(scores, k=kk, dim=1).indices
    correct = torch.arange(b, device=scores.device).unsqueeze(1)
    return float((topk == correct).any(dim=1).float().mean().item())


def retrieval_metrics(scores: torch.Tensor) -> dict[str, float]:
    pred = scores.argmax(dim=1)
    gt = torch.arange(scores.size(0), device=scores.device)
    acc = float((pred == gt).float().mean().item())
    hr5 = hr_at(scores, 5)
    hr10 = hr_at(scores, 10)
    hr20 = hr_at(scores, 20)
    return {
        "acc": acc,
        "hr5": hr5,
        "hr10": hr10,
        "hr20": hr20,
        "r5_at_10": hr5 / hr10 if hr10 > 1e-12 else 0.0,
        "r10_at_20": hr10 / hr20 if hr20 > 1e-12 else 0.0,
        "valid_loss_ce": float(torch.nn.functional.cross_entropy(scores, gt.long()).item()),
    }
