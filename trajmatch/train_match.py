from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from trajmatch.cache import TailCache, build_cache
from trajmatch.config import HYDRA_TAIL, K_SELECT, MATCH_DIM, MATCH_LR, pairs_path
from trajmatch.diffusion.ddpm import load_city
from trajmatch.match import PairMatcher, bce_pairs, retrieval_logits
from trajmatch.metrics import retrieval_metrics
from trajmatch.pairs import heads_xy, load_pairs, split_pairs


class _Set(Dataset):
    def __init__(self, pairs, indices):
        self.pairs = pairs
        self.indices = indices

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pre, post = heads_xy(self.pairs[i])
        return int(self.indices[i]), pre, post


def _collate(batch):
    idx, pre, post = zip(*batch)
    return torch.tensor(idx), torch.stack(pre), torch.stack(post)


def _valid_scores(model, info_pre, info_post, hydra_pre, hydra_post, batch, device):
    k = info_pre.size(0)
    scores = torch.empty((k, k), device=device)
    post_info = info_post.to(device)
    post_hydra = hydra_post.to(device)
    done = 0
    step = max(1, batch)
    while done < k:
        rows = slice(done, min(done + step, k))
        logits = retrieval_logits(
            model,
            info_pre[rows].to(device),
            post_info,
            hydra_pre[rows].to(device),
            post_hydra,
        )
        scores[rows] = logits
        done = rows.stop
    return scores


def _eval(model, info_pre, info_post, hydra_pre, hydra_post, frac, passes, seed, batch, device):
    n = info_pre.size(0)
    k = max(1, min(n, int(round(n * frac))))
    acc = hr5 = hr10 = hr20 = ce = 0.0
    model.eval()
    for rep in range(passes):
        g = torch.Generator().manual_seed(seed + rep * 7919)
        pick = torch.randperm(n, generator=g)[:k]
        scores = _valid_scores(
            model,
            info_pre[pick],
            info_post[pick],
            hydra_pre[pick],
            hydra_post[pick],
            batch,
            device,
        )
        stats = retrieval_metrics(scores)
        acc += stats["acc"]
        hr5 += stats["hr5"]
        hr10 += stats["hr10"]
        hr20 += stats["hr20"]
        ce += stats["valid_loss_ce"]
    passes = float(passes)
    return {"acc": acc / passes, "hr5": hr5 / passes, "hr10": hr10 / passes, "hr20": hr20 / passes, "valid_loss_ce": ce / passes}


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pairs = load_pairs(args.pairs_pkl, args.max_pairs)
    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = args.cache or os.path.join(args.out_dir, "tails.pt")
    if args.rebuild_cache or not os.path.exists(cache_path):
        pre, post = load_city(args.city, device)
        if args.diff_pre:
            pre.unet.load_state_dict(torch.load(args.diff_pre, map_location=device), strict=True)
        if args.diff_post:
            post.unet.load_state_dict(torch.load(args.diff_post, map_location=device), strict=True)
        build_cache(pairs, pre, post, hydra_tail=args.hydra_tail, batch=args.cache_batch, device=device).save(cache_path)
        del pre, post
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    train_pairs, valid_pairs, train_idx, valid_idx = split_pairs(pairs, args.valid_ratio, args.seed)
    cache = TailCache.load(cache_path)
    if cache.hydra_pre.size(0) != len(pairs):
        raise ValueError("tail cache rows do not match the pair file; rebuild the cache")
    use = cache.hydra_pre[:, : args.k_select], cache.hydra_post[:, : args.k_select]

    valid_pre = torch.stack([heads_xy(item)[0] for item in valid_pairs])
    valid_post = torch.stack([heads_xy(item)[1] for item in valid_pairs])
    vg = torch.tensor(valid_idx)
    valid_hpre = use[0][vg]
    valid_hpost = use[1][vg]

    model = PairMatcher(dim=args.match_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(_Set(train_pairs, train_idx), batch_size=args.batch, shuffle=True, collate_fn=_collate)
    best = -1.0
    stall = 0
    best_path = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for idxs, info_pre, info_post in tqdm(loader, desc=f"match {epoch}"):
            info_pre = info_pre.to(device)
            info_post = info_post.to(device)
            hydra_pre = use[0][idxs].to(device)
            hydra_post = use[1][idxs].to(device)
            if info_pre.size(0) < 2:
                continue
            if args.loss == "bb_ce":
                logits = retrieval_logits(model, info_pre, info_post, hydra_pre, hydra_post)
                loss = F.cross_entropy(logits, torch.arange(logits.size(0), device=device))
            else:
                loss = bce_pairs(model, info_pre, info_post, hydra_pre, hydra_post)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        stats = _eval(
            model, valid_pre, valid_post, valid_hpre, valid_hpost,
            args.valid_frac, args.valid_passes, args.seed + epoch, args.valid_batch, device,
        )
        print(f"epoch {epoch} loss {sum(losses) / max(1, len(losses)):.4f} acc {stats['acc']:.4f} hr5 {stats['hr5']:.4f}", flush=True)
        if stats["acc"] > best:
            best = stats["acc"]
            stall = 0
            best_path = os.path.join(args.out_dir, f"best_match_ep{epoch}_acc{best:.4f}.pth")
            torch.save(model.state_dict(), best_path)
        else:
            stall += 1
            if args.early_stop > 0 and stall >= args.early_stop:
                break
    print(f"best {best_path} acc {best:.4f}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build tail cache and train the pair matcher.")
    p.add_argument("--city", required=True, choices=("real_UAV", "chengdu"))
    p.add_argument("--pairs-pkl", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cache", default=None)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--diff-pre", default=None)
    p.add_argument("--diff-post", default=None)
    p.add_argument("--hydra-tail", type=int, default=HYDRA_TAIL)
    p.add_argument("--k-select", type=int, default=K_SELECT)
    p.add_argument("--cache-batch", type=int, default=16)
    p.add_argument("--match-dim", type=int, default=MATCH_DIM)
    p.add_argument("--loss", choices=("bb_ce", "bce"), default="bb_ce")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--valid-batch", type=int, default=10)
    p.add_argument("--lr", type=float, default=MATCH_LR)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--early-stop", type=int, default=10)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--valid-frac", type=float, default=0.15)
    p.add_argument("--valid-passes", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    if not args.pairs_pkl:
        args.pairs_pkl = str(pairs_path(args.city))
    return args


def main(argv=None):
    train(parse_args(argv))


if __name__ == "__main__":
    main()
