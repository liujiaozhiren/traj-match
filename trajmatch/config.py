from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HEAD_LEN = 8
TAIL_LEN = 4
HYDRA_TAIL = 64
K_SELECT = 16

DIFF_STEPS = 200
BETA_START = 1e-4
BETA_END = 0.05
DIFF_LR = 1e-6
UNET_CH = 512
LAMBDA_X0 = 0.0

MATCH_DIM = 128
MATCH_HEADS = 4
MATCH_LAYERS = 4
MATCH_LR = 1e-5

SELECTOR_DIM = 32
SELECTOR_LR = 1e-4
SELECTOR_ALPHA = 1.0
SELECTOR_LAMBDA = 1.0

ORIGIN_LON = 114.057868
ORIGIN_LAT = 22.543099

CITIES = ("real_UAV", "chengdu")


def city_name(city: str) -> str:
    raw = str(city).strip()
    name = next((item for item in CITIES if item.lower() == raw.lower()), None)
    if name is None:
        raise ValueError(f"city must be one of {CITIES}, got {city!r}")
    return name


def weight_paths(city: str) -> tuple[Path, Path]:
    folder = ROOT / "weights" / city_name(city)
    return folder / "diff_pre.pth", folder / "diff_post.pth"


def pairs_path(city: str) -> Path:
    return ROOT / "data" / city_name(city) / "pairs.pkl"
