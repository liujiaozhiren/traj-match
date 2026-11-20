from __future__ import annotations

import math
from typing import Sequence

from trajmatch.config import ORIGIN_LAT, ORIGIN_LON


def _order_lon_lat(a: float, b: float) -> tuple[float, float]:
    d_keep = abs(a - ORIGIN_LON) + abs(b - ORIGIN_LAT)
    d_swap = abs(a - ORIGIN_LAT) + abs(b - ORIGIN_LON)
    if d_keep <= d_swap:
        return a, b
    return b, a


def lonlat_to_xy(lon: float, lat: float) -> tuple[float, float]:
    lat0 = math.radians(ORIGIN_LAT)
    x = (lon - ORIGIN_LON) * (111320.0 * math.cos(lat0))
    y = (lat - ORIGIN_LAT) * 110540.0
    return x, y


def points_to_xy(pts: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    out = []
    for p in pts:
        lon, lat = _order_lon_lat(float(p[1]), float(p[2]))
        out.append(lonlat_to_xy(lon, lat))
    return out
