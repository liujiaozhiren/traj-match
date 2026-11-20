from __future__ import annotations

import argparse
import json

import torch

from trajmatch.cache import TailCache
from trajmatch.config import K_SELECT, MATCH_DIM, pairs_path
from trajmatch.match import PairMatcher
from trajmatch.pairs import heads_xy, load_pairs, split_pairs
from trajmatch.selector import SelectorState, TailSelector
from trajmatch.train_selector import _eval_selector


def main(argv=None):
    p = argparse.ArgumentParser(description="Score a frozen matcher, with or without a tail selector.")
    p.add_argument("--city", default="real_UAV", choices=("real_UAV", "chengdu"))
    p.add_argument("--pairs-pkl", default=None)
    p.add_argument("--cache", required=True)
    p.add_argument("--match-ckpt", required=True)
    p.add_argument("--selector-ckpt", default=None)
    p.add_argument("--out-json", default=None)
    p.add_argument("--k-select", type=int, default=K_SELECT)
    p.add_argument("--hydra-tail", type=int, default=64)
    p.add_argument("--match-dim", type=int, default=MATCH_DIM)
    p.add_argument("--alpha", type=float, default=1.0, help="1 keeps the noisy head; 0 uses the label head")
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--valid-ratio", type=float, default=0.1)
    p.add_argument("--valid-frac", type=float, default=0.15)
    p.add_argument("--valid-passes", type=int, default=3)
    p.add_argument("--valid-batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)
    if not args.pairs_pkl:
        args.pairs_pkl = str(pairs_path(args.city))

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pairs = load_pairs(args.pairs_pkl, args.max_pairs)
    _, valid_pairs, _, valid_idx = split_pairs(pairs, args.valid_ratio, args.seed)
    cache = TailCache.load(args.cache)
    match = PairMatcher(dim=args.match_dim).to(device)
    match.load_state_dict(torch.load(args.match_ckpt, map_location=device), strict=True)
    selector = None
    if args.selector_ckpt:
        selector = TailSelector(k_select=args.k_select, device=device)
        selector.load_state(SelectorState.load(args.selector_ckpt))
    info_pre = torch.stack([heads_xy(item, alpha=args.alpha)[0] for item in valid_pairs])
    info_post = torch.stack([heads_xy(item, alpha=args.alpha)[1] for item in valid_pairs])
    vg = torch.tensor(valid_idx)
    stats = _eval_selector(
        match,
        selector if selector is not None else _FirstK(args.k_select),
        info_pre,
        info_post,
        cache.hydra_pre[vg, : args.hydra_tail],
        cache.hydra_post[vg, : args.hydra_tail],
        args.k_select,
        args.valid_frac,
        args.valid_passes,
        args.seed,
        args.valid_batch,
        device,
    )
    print(json.dumps(stats, indent=2))
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)


class _FirstK:
    def __init__(self, k):
        self.k = k

    def choose(self, info_pre, info_post, hydra_pre, hydra_post):
        return hydra_pre[:, : self.k], hydra_post[:, : self.k], None, None


if __name__ == "__main__":
    main()
