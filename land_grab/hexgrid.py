"""Hexagon grid: Uber's H3 grid at resolution 8.

The Land Grab challenge scores hexagons by their H3 cell ID, so this module uses
the real H3 grid instead of drawing hexagons of its own. At resolution 8 a cell
is roughly 0.8 km across (flat side to flat side). A cell ID is a string such as
``"88194ad147fffff"``.

Uses the version 4 API of the ``h3`` package: ``latlng_to_cell``,
``cell_to_boundary``, ``cell_to_latlng``, ``grid_disk``, ``grid_distance``.

Distances are worked out in metres on the British National Grid (EPSG:27700),
then converted back to WGS84 (lat/lon) for output. A cell's boundary comes from
H3 itself (``cell_to_boundary``), so the 25 m safe margin is measured to the
real edge of the real cell. Note that H3 also has 12 pentagons in the world;
they have 5 neighbours instead of 6 (none are near the UK).
"""

import math
import re
from collections.abc import Iterable
from functools import lru_cache

import h3
from pyproj import Transformer
from shapely.geometry import Point, Polygon

RESOLUTION = 8

# always_xy=True keeps (lon, lat) / (easting, northing) ordering on both sides.
_TO_BNG = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
_TO_WGS84 = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

HexId = str

# One circle of cells is at least this far across, so ``k`` rings reach at least ``k * this`` metres.
_MIN_CELL_SPACING_M = 600.0


class InvalidCellError(ValueError):
    """Text that was meant to be an H3 cell ID is not one (the message is written for people)."""


def latlon_to_hex(lat: float, lon: float) -> HexId:
    """Return the H3 cell ID (resolution 8) of the cell containing a lat/lon point."""
    return h3.latlng_to_cell(lat, lon, RESOLUTION)


def bng_to_hex(easting: float, northing: float) -> HexId:
    """Return the H3 cell ID of the cell containing a British National Grid point."""
    lon, lat = _TO_WGS84.transform(easting, northing)
    return latlon_to_hex(lat, lon)


def hex_centre(hex_id: HexId) -> tuple[float, float]:
    """The cell's centre as (lat, lon)."""
    lat, lon = h3.cell_to_latlng(hex_id)
    return lat, lon


def hex_centre_bng(hex_id: HexId) -> tuple[float, float]:
    """The cell's centre as (easting, northing) in metres."""
    lat, lon = hex_centre(hex_id)
    return _TO_BNG.transform(lon, lat)


def hex_boundary(hex_id: HexId) -> list[tuple[float, float]]:
    """The cell's corners as (lat, lon) pairs, straight from ``h3.cell_to_boundary``."""
    return [(lat, lon) for lat, lon in h3.cell_to_boundary(hex_id)]


def hex_to_polygon(hex_id: HexId) -> Polygon:
    """Return the cell as a shapely Polygon in lat/lon (x=lon, y=lat)."""
    return Polygon([(lon, lat) for lat, lon in hex_boundary(hex_id)])


@lru_cache(maxsize=20_000)
def hex_polygon_bng(hex_id: HexId) -> Polygon:
    """The cell as a shapely Polygon in metres (x=easting, y=northing), made once per cell."""
    return Polygon([_TO_BNG.transform(lon, lat) for lat, lon in hex_boundary(hex_id)])


def depth_in_hex(x: float, y: float, hex_id: HexId) -> float:
    """Distance in metres from a BNG point to the nearest edge of the H3 cell.

    Positive inside the cell, negative outside. This is the GPS safety margin
    check: a route only counts a cell if it gets far enough inside it.
    """
    polygon = hex_polygon_bng(hex_id)
    point = Point(x, y)
    distance = polygon.exterior.distance(point)
    return distance if polygon.contains(point) else -distance


@lru_cache(maxsize=100_000)
def neighbours(hex_id: HexId) -> tuple[HexId, ...]:
    """The cells that share an edge with this one (6, or 5 for a pentagon), in a fixed order."""
    return tuple(sorted(cell for cell in h3.grid_disk(hex_id, 1) if cell != hex_id))


def hex_distance(a: HexId, b: HexId) -> int:
    """Number of steps between two cells (``h3.grid_distance``)."""
    return h3.grid_distance(a, b)


def surrounding_hexes(hex_ids: Iterable[HexId], rings: int) -> set[HexId]:
    """Return the cells within ``rings`` steps of any cell in ``hex_ids``, including them."""
    area: set[HexId] = set()
    for hex_id in hex_ids:
        area.update(h3.grid_disk(hex_id, rings))
    return area


def hexes_within_radius(lat: float, lon: float, radius_m: float) -> set[HexId]:
    """Return ids of all cells that intersect the circle of radius_m around a point."""
    if radius_m < 0:
        raise ValueError(f"radius_m must not be negative, got {radius_m}")

    centre = Point(*_TO_BNG.transform(lon, lat))
    rings = math.ceil(radius_m / _MIN_CELL_SPACING_M) + 1
    return {
        hex_id
        for hex_id in h3.grid_disk(latlon_to_hex(lat, lon), rings)
        if centre.distance(hex_polygon_bng(hex_id)) <= radius_m
    }


# --- cell IDs typed by people -----------------------------------------------------------


def is_valid_hex(text: str) -> bool:
    """True if ``text`` is a valid H3 cell ID at resolution 8."""
    try:
        return bool(h3.is_valid_cell(text)) and h3.get_resolution(text) == RESOLUTION
    except Exception:  # the h3 package raises its own errors for text it cannot read
        return False


def parse_cells(text: str | Iterable[str] | None) -> frozenset[HexId]:
    """Turn pasted cell IDs (comma-separated text, or a list of strings) into a set of IDs.

    Spaces, new lines and upper case are tolerated, and empty pieces are ignored,
    so ``"88194ad147fffff, 88194ad141fffff,"`` works. Raises InvalidCellError,
    with a message that names the bad pieces, if any is not a resolution-8 cell.
    """
    if text is None:
        return frozenset()
    pieces = re.split(r"[,\s;]+", text) if isinstance(text, str) else [str(p) for p in text]
    cells = [piece.strip().lower() for piece in pieces if piece.strip()]
    bad = [cell for cell in cells if not is_valid_hex(cell)]
    if bad:
        shown = ", ".join(f'"{cell}"' for cell in bad[:3]) + (" and more" if len(bad) > 3 else "")
        raise InvalidCellError(
            f"These don't look like H3 cell IDs at resolution {RESOLUTION}: {shown}. "
            f'A cell ID looks like "88194ad147fffff"; separate several with commas.'
        )
    return frozenset(cells)
