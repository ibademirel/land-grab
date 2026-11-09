"""Work out which H3 cells a GPS track passes through."""

import math

from land_grab.hexgrid import _TO_BNG, HexId, bng_to_hex, depth_in_hex

MAX_SEGMENT_M = 20.0
DEFAULT_MIN_DEPTH_M = 25.0  # GPS safety margin: how far inside a hexagon the route must get


def _densify_bng(
    points: list[tuple[float, float]], max_segment_m: float = MAX_SEGMENT_M
) -> list[tuple[float, float]]:
    """Project (lat, lon) points to BNG and insert points so no segment exceeds max_segment_m.

    Interpolation is linear in metres (BNG), so the added points lie on the
    straight line between the originals. The original points are all kept.
    """
    projected = [_TO_BNG.transform(lon, lat) for lat, lon in points]
    if not projected:
        return []

    dense = [projected[0]]
    for (x0, y0), (x1, y1) in zip(projected, projected[1:]):
        steps = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / max_segment_m))
        for i in range(1, steps + 1):
            t = i / steps
            dense.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return dense


def _meets_margin(depth_m: float, min_depth_m: float) -> bool:
    """True if a point at ``depth_m`` is far enough inside; a zero margin accepts everything."""
    return min_depth_m <= 0 or depth_m >= min_depth_m


def _validate_min_depth(min_depth_m: float) -> None:
    if min_depth_m < 0:
        raise ValueError(f"min_depth_m must not be negative, got {min_depth_m}")


def hexes_for_track(
    points: list[tuple[float, float]],
    min_depth_m: float = DEFAULT_MIN_DEPTH_M,
) -> set[HexId]:
    """Return the ids of the hexagons a track of (lat, lon) points properly passes through.

    The track is densified to at most 20 m between samples before each sample
    is assigned to a hexagon. A hexagon counts only if at least one sample is
    at least ``min_depth_m`` inside it (distance to the nearest cell edge),
    a safety margin against GPS error for routes that only clip an edge or
    corner. ``min_depth_m=0`` counts every hexagon a sample falls in. An empty
    track returns an empty set.
    """
    _validate_min_depth(min_depth_m)
    found: set[HexId] = set()
    for x, y in _densify_bng(points):
        hex_id = bng_to_hex(x, y)
        if hex_id not in found and _meets_margin(depth_in_hex(x, y, hex_id), min_depth_m):
            found.add(hex_id)
    return found


def risky_hexes(
    points: list[tuple[float, float]],
    min_depth_m: float = DEFAULT_MIN_DEPTH_M,
) -> set[HexId]:
    """Return the hexagons the track touches but that do not meet the ``min_depth_m`` margin."""
    return hexes_for_track(points, 0) - hexes_for_track(points, min_depth_m)


def count_unique_hexes(
    points: list[tuple[float, float]],
    min_depth_m: float = DEFAULT_MIN_DEPTH_M,
) -> int:
    """Return the number of distinct hexagons a track properly passes through (see hexes_for_track)."""
    return len(hexes_for_track(points, min_depth_m))
