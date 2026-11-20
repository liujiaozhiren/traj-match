from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader, Dataset

from trajmatch.config import DIFF_LR, DIFF_STEPS, LAMBDA_X0, pairs_path
from trajmatch.diffusion.ddpm import ConditionalDDPM, load_city
from trajmatch.pairs import diffusion_example, load_pairs, split_pairs


class _DiffSet(Dataset):
    def __init__(self, pairs):
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return diffusion_example(self.pairs[idx])


def _collate(batch):
    cols = list(zip(*batch))
    return tuple(torch.stack(col) for col in cols)


def rmse_meters(pred, target) -> float:
    if pred.shape[1] != 2:
        pred = pred.transpose(1, 2)
        target = target.transpose(1, 2)
    per = ((pred - target) ** 2).sum(dim=1).mean(dim=1).sqrt()
    return float(per.mean().item())


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pairs = load_pairs(args.pairs_pkl, args.max_pairs)
    train_pairs, valid_pairs, _, _ = split_pairs(pairs, args.valid_ratio, args.seed)
    train_loader = DataLoader(_DiffSet(train_pairs), batch_size=args.batch, shuffle=True, collate_fn=_collate)
    valid_loader = DataLoader(_DiffSet(valid_pairs), batch_size=args.eval_batch, shuffle=False, collate_fn=_collate)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.from_scratch:
        pre = ConditionalDDPM(steps=args.steps, lr=args.lr).to(device)
        post = ConditionalDDPM(steps=args.steps, lr=args.lr).to(device)
    else:
        pre, post = load_city(args.city, device, steps=args.steps, lr=args.lr)
        if args.ckpt_pre:
            pre.unet.load_state_dict(torch.load(args.ckpt_pre, map_location=device), strict=True)
        if args.ckpt_post:
            post.unet.load_state_dict(torch.load(args.ckpt_post, map_location=device), strict=True)

    best_pre = float("inf")
    best_post = float("inf")
    stall = 0
    for epoch in range(1, args.epochs + 1):
        pre.train()
        post.train()
        losses = []
        for info_pre, lab_pre, info_post, lab_post in train_loader:
            info_pre, lab_pre = info_pre.to(device), lab_pre.to(device)
            info_post, lab_post = info_post.to(device), lab_post.to(device)
            loss_pre, _ = pre.learn(torch.cat([info_pre, lab_pre], dim=2), lam=args.lambda_x0)
            loss_post, _ = post.learn(torch.cat([info_post, lab_post], dim=2), lam=args.lambda_x0)
            losses.append(0.5 * (loss_pre + loss_post))

        pre.eval()
        post.eval()
        err_pre, err_post = [], []
        with torch.no_grad():
            for info_pre, lab_pre, info_post, lab_post in valid_loader:
                info_pre, lab_pre = info_pre.to(device), lab_pre.to(device)
                info_post, lab_post = info_post.to(device), lab_post.to(device)
                err_pre.append(rmse_meters(pre.sample(info_pre), lab_pre))
                err_post.append(rmse_meters(post.sample(info_post), lab_post))
        rmse_pre = sum(err_pre) / max(1, len(err_pre))
        rmse_post = sum(err_post) / max(1, len(err_post))
        improved = rmse_pre < best_pre or rmse_post < best_post
        if rmse_pre < best_pre:
            best_pre = rmse_pre
            torch.save(pre.unet.state_dict(), os.path.join(args.out_dir, "best_pre.pth"))
        if rmse_post < best_post:
            best_post = rmse_post
            torch.save(post.unet.state_dict(), os.path.join(args.out_dir, "best_post.pth"))
        print(
            f"epoch {epoch} loss {sum(losses) / max(1, len(losses)):.6f} "
            f"rmse_pre {rmse_pre:.3f} rmse_post {rmse_post:.3f}",
            flush=True,
        )
        stall = 0 if improved else stall + 1
        if args.early_stop > 0 and stall >= args.early_stop:
            break

    with open(os.path.join(args.out_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best_pre_rmse_m": best_pre, "best_post_rmse_m": best_post, "city": args.city}, f, indent=2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune the pre/post tail diffusion models.")
    p.add_argument("--city", required=True, choices=("real_UAV", "chengdu"))
    p.add_argument("--pairs-pkl", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--from-scratch", action="store_true")
    p.add_argument("--ckpt-pre", default=None)
    p.add_argument("--ckpt-post", default=None)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=DIFF_LR)
    p.add_argument("--steps", type=int, default=DIFF_STEPS)
    p.add_argument("--lambda-x0", type=float, default=LAMBDA_X0)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--early-stop", type=int, default=20)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    if not args.pairs_pkl:
        args.pairs_pkl = str(pairs_path(args.city))
    return args


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()
