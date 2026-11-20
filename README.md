# MMHM

This repository is the open-source implementation of *MMHM: Beyond Observation-Space Matching for Cross-Sensor Trajectory Association*.

The supplementary material is [Supplementary Material for MMHM.pdf](Supplementary%20Material%20for%20MMHM.pdf).

## Data

We release the real_UAV dataset. It contains two years of drone trajectories collected at base-station handovers. Each record aligns a drone flight trajectory with the trajectory observed by the base stations. The file is `data/real_UAV/pairs.pkl`, selected with `--city real_UAV`.

For user privacy, identity information and other identifying fields have been removed from the released data. Trajectory coordinates have also been shifted onto one shared geographic location. The released data uses the origin defined in `trajmatch/config.py`, so the original collection locations are not retained.

Each file is a pickle containing a list of 7-tuples. Each item is `(_, A, B, L_A, L_B, _, _)`. Training and evaluation use `A`, `B`, `L_A`, and `L_B`. `A` / `B` are the observations from the two sides, and `L_A` / `L_B` are the corresponding complete trajectories. A point is `[t, lon, lat]` or `[t, lat, lon]`. To use the Chengdu data, switch to `--city chengdu`, which reads `data/chengdu/pairs.pkl`.

## Requirements

Python 3.10.

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

## Training order

Run the commands from the repository root with `PYTHONPATH=.`. The diffusion model uses an observation length of 8 and a generated length of 4. Training proceeds in this order:

1. Pretrain the diffusion model.
2. Fine-tune the diffusion model.
3. Train the matcher.
4. Train the selector, then evaluate.

Pretraining the diffusion model takes a long time. The pretrained weights are included so this step can be skipped:

- `weights/real_UAV/diff_pre.pth`, `weights/real_UAV/diff_post.pth`
- `weights/chengdu/diff_pre.pth`, `weights/chengdu/diff_post.pth`

`diff_pre.pth` and `diff_post.pth` are the two sides of a trajectory. To run pretraining instead of using these weights, add `--from-scratch` to `train_diffusion`. The commands below start from the released pretrained weights, fine-tune them, and pass the fine-tuned checkpoints to the matcher.

```bash
export PYTHONPATH=.

python -m trajmatch.train_diffusion --city real_UAV --out-dir runs/uav_diff
python -m trajmatch.train_match --city real_UAV --diff-pre runs/uav_diff/best_pre.pth --diff-post runs/uav_diff/best_post.pth --out-dir runs/uav_match
python -m trajmatch.train_selector --city real_UAV --cache runs/uav_match/tails.pt --match-ckpt runs/uav_match/best_match_*.pth --out-dir runs/uav_select
python -m trajmatch.evaluate --city real_UAV --cache runs/uav_match/tails.pt --match-ckpt runs/uav_match/best_match_*.pth --selector-ckpt runs/uav_select/best_selector_*.pt
```
