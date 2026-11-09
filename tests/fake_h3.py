"""A small stand-in for the ``h3`` package, used ONLY when the real one is not installed.

The real ``h3`` package could not be installed in the sandbox this project was
built in, so ``conftest.py`` puts this stand-in in its place to let the rest of
the test suite run. It copies the version 4 API that land_grab uses
(``latlng_to_cell``, ``cell_to_boundary``, ``cell_to_latlng``, ``grid_disk``,
``grid_distance``, ``get_resolution``, ``is_valid_cell``) on a plain hexagon grid
over the British National Grid, with cells 800 m flat to flat (about the size
of a real resolution-8 cell).

It is NOT H3: the cell IDs are made up, there are no pentagons and the cell
shapes are perfect hexagons. Everything that depends on how real H3 behaves is
checked by ``test_h3_benchmark.py``, which is skipped when the real package is
missing. With the real package installed, this file is never used.
"""

import math
import re

from pyproj import Transformer

SIZE_M = 800.0
_TO_BNG = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
_NEIGHBOURS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
_PATTERN = re.compile(r"^88fa([0-9a-f]{4})([0-9a-f]{4})fff$")
_OFFSET = 1 << 15


def _to_id(q: int, r: int) -> str:
    return f"88fa{q + _OFFSET:04x}{r + _OFFSET:04x}fff"


def _from_id(cell: str) -> tuple[int, int]:
    match = _PATTERN.match(cell)
    if not match:
        raise ValueError(f"not a cell: {cell!r}")
    return int(match.group(1), 16) - _OFFSET, int(match.group(2), 16) - _OFFSET


def _round_axial(q: float, r: float) -> tuple[int, int]:
    x, z = q, r
    y = -x - z
    rx, ry, rz = round(x), round(y), round(z)
    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return int(rx), int(rz)


def _centre_bng(q: int, r: int) -> tuple[float, float]:
    return SIZE_M * (q + r / 2), SIZE_M * math.sqrt(3) / 2 * r


def latlng_to_cell(lat: float, lng: float, res: int) -> str:
    assert res == 8, "the stand-in only does resolution 8"
    easting, northing = _TO_BNG.transform(lng, lat)
    r = northing / (SIZE_M * math.sqrt(3) / 2)
    q = easting / SIZE_M - r / 2
    return _to_id(*_round_axial(q, r))


def cell_to_latlng(cell: str) -> tuple[float, float]:
    lon, lat = _TO_WGS84.transform(*_centre_bng(*_from_id(cell)))
    return lat, lon


def cell_to_boundary(cell: str) -> tuple[tuple[float, float], ...]:
    cx, cy = _centre_bng(*_from_id(cell))
    radius = SIZE_M / math.sqrt(3)
    corners = []
    for i in range(6):
        angle = math.radians(30 + 60 * i)
        lon, lat = _TO_WGS84.transform(cx + radius * math.cos(angle), cy + radius * math.sin(angle))
        corners.append((lat, lon))
    return tuple(corners)


def grid_disk(cell: str, k: int) -> list[str]:
    q, r = _from_id(cell)
    return [
        _to_id(q + dq, r + dr)
        for dq in range(-k, k + 1)
        for dr in range(max(-k, -dq - k), min(k, -dq + k) + 1)
    ]


def grid_distance(a: str, b: str) -> int:
    (qa, ra), (qb, rb) = _from_id(a), _from_id(b)
    dq, dr = qa - qb, ra - rb
    return max(abs(dq), abs(dr), abs(dq + dr))


def get_resolution(cell: str) -> int:
    _from_id(cell)
    return 8


def is_valid_cell(cell) -> bool:
    return isinstance(cell, str) and bool(_PATTERN.match(cell))
