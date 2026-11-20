from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from trajmatch.cache import TailCache
from trajmatch.config import K_SELECT, MATCH_DIM, SELECTOR_LR, pairs_path
from trajmatch.match import PairMatcher, retrieval_logits
from trajmatch.metrics import retrieval_metrics
from trajmatch.pairs import heads_xy, load_pairs, split_pairs
from trajmatch.selector import TailSelector, normalize_reward
from trajmatch.train_match import _Set, _collate, _valid_scores


def _arm_bce(model, info_pre, info_post, hydra_pre, hydra_post, label: float):
    b, k, length, _ = hydra_pre.shape
    logits = model(
        info_pre.repeat_interleave(k, dim=0),
        info_post.repeat_interleave(k, dim=0),
        hydra_pre.reshape(b * k, 1, length, 2),
        hydra_post.reshape(b * k, 1, length, 2),
    )
    target = torch.full_like(logits, float(label))
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none")


def _reward(model, info_pre, info_post, base_pre, base_post, sel_pre, sel_post, label: float, mode: str):
    k = sel_pre.size(1)
    b = info_pre.size(0)
    if mode == "bb_ce" and label > 0.5:
        base = F.cross_entropy(retrieval_logits(model, info_pre, info_post, base_pre, base_post), torch.arange(b, device=info_pre.device))
        chosen = F.cross_entropy(retrieval_logits(model, info_pre, info_post, sel_pre, sel_post), torch.arange(b, device=info_pre.device))
        return torch.full((b * k,), float((base - chosen) / (b * k)), device=info_pre.device)
    base_loss = _arm_bce(model, info_pre, info_post, base_pre, base_post, label).view(b, k)
    sel_loss = _arm_bce(model, info_pre, info_post, sel_pre, sel_post, label).view(b, k)
    return (base_loss.mean(dim=1, keepdim=True) - sel_loss).reshape(-1)


def train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pairs = load_pairs(args.pairs_pkl, args.max_pairs)
    train_pairs, valid_pairs, train_idx, valid_idx = split_pairs(pairs, args.valid_ratio, args.seed)
    cache = TailCache.load(args.cache)
    if cache.hydra_pre.size(0) != len(pairs):
        raise ValueError("tail cache rows do not match the pair file")
    hydra = cache.hydra_pre[:, : args.hydra_tail], cache.hydra_post[:, : args.hydra_tail]
    os.makedirs(args.out_dir, exist_ok=True)

    match = PairMatcher(dim=args.match_dim).to(device)
    match.load_state_dict(torch.load(args.match_ckpt, map_location=device), strict=True)
    match.eval()
    for p in match.parameters():
        p.requires_grad = False

    selector = TailSelector(k_select=args.k_select, lr=args.lr, device=device)
    loader = DataLoader(_Set(train_pairs, train_idx), batch_size=args.batch, shuffle=True, collate_fn=_collate)
    valid_pre = torch.stack([heads_xy(item)[0] for item in valid_pairs])
    valid_post = torch.stack([heads_xy(item)[1] for item in valid_pairs])
    vg = torch.tensor(valid_idx)
    best = -1.0
    stall = 0
    for epoch in range(1, args.epochs + 1):
        for idxs, info_pre, info_post in loader:
            if info_pre.size(0) < 2:
                continue
            info_pre = info_pre.to(device)
            info_post = info_post.to(device)
            hpre = hydra[0][idxs].to(device)
            hpost = hydra[1][idxs].to(device)
            shift = int(torch.randint(1, info_pre.size(0), (1,), device=device).item())
            perm = (torch.arange(info_pre.size(0), device=device) + shift) % info_pre.size(0)
            for label, ip, io, hp, ho in (
                (1.0, info_pre, info_post, hpre, hpost),
                (0.0, info_pre, info_post[perm], hpre, hpost[perm]),
            ):
                base_pre, base_post = hp[:, : args.k_select], ho[:, : args.k_select]
                sel_pre, sel_post, _, _ = selector.choose(ip, io, hp, ho)
                with torch.no_grad():
                    reward = _reward(match, ip, io, base_pre, base_post, sel_pre, sel_post, label, args.reward)
                selector.update(normalize_reward(reward, clip=args.reward_clip))

        stats = _eval_selector(
            match, selector, valid_pre, valid_post, hydra[0][vg], hydra[1][vg],
            args.k_select, args.valid_frac, args.valid_passes, args.seed + epoch, args.valid_batch, device,
        )
        print(f"epoch {epoch} acc {stats['acc']:.4f} vanilla {stats['vanilla_acc']:.4f}", flush=True)
        if stats["acc"] > best:
            best = stats["acc"]
            stall = 0
            selector.state().save(os.path.join(args.out_dir, f"best_selector_ep{epoch}_acc{best:.4f}.pt"))
        else:
            stall += 1
            if args.early_stop > 0 and stall >= args.early_stop:
                break


@torch.no_grad()
def _eval_selector(match, selector, info_pre, info_post, hydra_pre, hydra_post, k_select, frac, passes, seed, batch, device):
    n = info_pre.size(0)
    k = max(1, min(n, int(round(n * frac))))
    acc = vacc = 0.0
    match.eval()
    for rep in range(passes):
        g = torch.Generator().manual_seed(seed + rep * 7919)
        pick = torch.randperm(n, generator=g)[:k]
        scores = _select_scores(match, selector, info_pre[pick], info_post[pick], hydra_pre[pick], hydra_post[pick], batch, device)
        vanilla = _valid_scores(
            match,
            info_pre[pick],
            info_post[pick],
            hydra_pre[pick][:, :k_select],
            hydra_post[pick][:, :k_select],
            batch,
            device,
        )
        acc += retrieval_metrics(scores)["acc"]
        vacc += retrieval_metrics(vanilla)["acc"]
    return {"acc": acc / passes, "vanilla_acc": vacc / passes}


@torch.no_grad()
def _select_scores(match, selector, info_pre, info_post, hydra_pre, hydra_post, batch, device):
    k = info_pre.size(0)
    scores = torch.empty((k, k), device=device)
    post_info = info_post.to(device)
    post_hydra = hydra_post.to(device)
    done = 0
    while done < k:
        rows = slice(done, min(done + batch, k))
        q_info = info_pre[rows].to(device)
        q_hydra = hydra_pre[rows].to(device)
        b1, b2 = q_info.size(0), post_info.size(0)
        ip = q_info.unsqueeze(1).expand(b1, b2, -1, -1).reshape(b1 * b2, q_info.size(1), 2)
        io = post_info.unsqueeze(0).expand(b1, b2, -1, -1).reshape(b1 * b2, post_info.size(1), 2)
        hp = q_hydra.unsqueeze(1).expand(b1, b2, -1, -1, -1).reshape(b1 * b2, q_hydra.size(1), q_hydra.size(2), 2)
        ho = post_hydra.unsqueeze(0).expand(b1, b2, -1, -1, -1).reshape(b1 * b2, post_hydra.size(1), post_hydra.size(2), 2)
        sp, so, _, _ = selector.choose(ip, io, hp, ho)
        scores[rows] = match(ip, io, sp, so).view(b1, b2)
        done = rows.stop
    return scores


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the tail selector on a frozen matcher.")
    p.add_argument("--city", default="real_UAV", choices=("real_UAV", "chengdu"))
    p.add_argument("--pairs-pkl", default=None)
    p.add_argument("--cache", required=True)
    p.add_argument("--match-ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--hydra-tail", type=int, default=64)
    p.add_argument("--k-select", type=int, default=K_SELECT)
    p.add_argument("--match-dim", type=int, default=MATCH_DIM)
    p.add_argument("--reward", choices=("bb_ce", "bce"), default="bb_ce")
    p.add_argument("--reward-clip", type=float, default=3.0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--valid-batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=SELECTOR_LR)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--early-stop", type=int, default=15)
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
