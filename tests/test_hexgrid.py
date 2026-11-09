"""Tests for land_grab.hexgrid (the H3 grid at resolution 8).

These must pass with the real ``h3`` package and with the stand-in in fake_h3.py, so they only
rely on things true for any grid of cells about 0.8 km across. How real H3 behaves is checked in
test_h3_benchmark.py.
"""

import pytest
from shapely.geometry import Point

from land_grab.hexgrid import (
    _TO_BNG,
    _TO_WGS84,
    RESOLUTION,
    InvalidCellError,
    bng_to_hex,
    depth_in_hex,
    hex_boundary,
    hex_centre,
    hex_centre_bng,
    hex_distance,
    hex_polygon_bng,
    hex_to_polygon,
    hexes_within_radius,
    is_valid_hex,
    latlon_to_hex,
    neighbours,
    parse_cells,
    surrounding_hexes,
)

# Buckingham Palace area, London
LAT, LON = 51.5010, -0.1416


def _offset_latlon(lat, lon, de, dn):
    """Move a lat/lon point by de metres east and dn metres north (in BNG)."""
    e, n = _TO_BNG.transform(lon, lat)
    new_lon, new_lat = _TO_WGS84.transform(e + de, n + dn)
    return new_lat, new_lon


HOME = latlon_to_hex(LAT, LON)


def test_resolution_is_8():
    assert RESOLUTION == 8


def test_cell_ids_are_strings():
    assert isinstance(HOME, str) and len(HOME) == 15


def test_the_centre_of_a_cell_maps_back_to_the_same_cell():
    for de, dn in [(0, 0), (3000, 0), (-5000, 7000), (20000, -10000)]:
        cell = latlon_to_hex(*_offset_latlon(LAT, LON, de, dn))
        assert latlon_to_hex(*hex_centre(cell)) == cell


def test_point_and_point_100m_from_the_centre_share_a_cell():
    # Start at the centre so a 100 m step cannot cross an edge (the cells are about 800 m across).
    lat, lon = hex_centre(HOME)
    assert latlon_to_hex(*_offset_latlon(lat, lon, 100, 0)) == HOME


def test_point_3km_away_is_in_a_different_cell():
    assert latlon_to_hex(*_offset_latlon(LAT, LON, 3000, 0)) != HOME


def test_bng_and_lat_lon_agree():
    e, n = _TO_BNG.transform(LON, LAT)
    assert bng_to_hex(e, n) == HOME


def test_boundary_comes_from_h3_and_has_six_corners():
    boundary = hex_boundary(HOME)
    assert len(boundary) == 6
    assert all(51 < lat < 52 and -1 < lon < 1 for lat, lon in boundary)


def test_polygon_contains_its_own_centre_and_the_point_used_to_find_it():
    lat, lon = hex_centre(HOME)
    assert hex_to_polygon(HOME).contains(Point(lon, lat))
    assert hex_to_polygon(HOME).contains(Point(LON, LAT))


def test_cell_is_roughly_a_kilometre_across():
    minx, miny, maxx, maxy = hex_polygon_bng(HOME).bounds
    assert 600 < maxx - minx < 1400 and 600 < maxy - miny < 1400
    assert 0.3e6 < hex_polygon_bng(HOME).area < 1.2e6  # square metres


def test_depth_is_positive_inside_zero_on_the_edge_negative_outside():
    cx, cy = hex_centre_bng(HOME)
    polygon = hex_polygon_bng(HOME)
    assert depth_in_hex(cx, cy, HOME) > 300  # the middle of a cell is far from every edge
    assert depth_in_hex(cx, cy, HOME) == pytest.approx(polygon.exterior.distance(Point(cx, cy)))
    corner_x, corner_y = polygon.exterior.coords[0]
    assert depth_in_hex(corner_x, corner_y, HOME) == pytest.approx(0, abs=1e-6)
    assert depth_in_hex(cx + 5000, cy, HOME) < 0


def test_depth_is_the_distance_to_the_cell_boundary_not_to_a_made_up_polygon():
    cx, cy = hex_centre_bng(HOME)
    polygon = hex_polygon_bng(HOME)
    # Walk from the centre towards an edge; the depth drops by the distance walked.
    (ax, ay), (bx, by) = polygon.exterior.coords[0], polygon.exterior.coords[1]
    mid = ((ax + bx) / 2, (ay + by) / 2)
    towards = (mid[0] - cx, mid[1] - cy)
    length = (towards[0] ** 2 + towards[1] ** 2) ** 0.5
    x, y = cx + towards[0] * (length - 20) / length, cy + towards[1] * (length - 20) / length
    assert depth_in_hex(x, y, HOME) == pytest.approx(20, abs=0.5)


def test_a_cell_has_six_neighbours_that_share_an_edge():
    ring = neighbours(HOME)
    assert len(ring) == 6 and HOME not in ring
    assert all(hex_distance(HOME, other) == 1 for other in ring)
    assert all(HOME in neighbours(other) for other in ring)
    assert neighbours(HOME) == tuple(sorted(ring))  # a fixed order, so plans are repeatable


def test_hex_distance():
    assert hex_distance(HOME, HOME) == 0
    far = latlon_to_hex(*_offset_latlon(LAT, LON, 5000, 0))
    assert hex_distance(HOME, far) >= 4


def test_surrounding_hexes_counts_the_rings():
    assert surrounding_hexes({HOME}, 0) == {HOME}
    assert len(surrounding_hexes({HOME}, 1)) == 7
    assert len(surrounding_hexes({HOME}, 2)) == 19
    assert surrounding_hexes({HOME}, 1) >= set(neighbours(HOME))


def test_hexes_within_radius_contains_the_start_cell_and_grows_with_the_radius():
    assert hexes_within_radius(LAT, LON, 0) == {HOME}
    small, big = hexes_within_radius(LAT, LON, 1000), hexes_within_radius(LAT, LON, 3000)
    assert HOME in small and small < big
    assert all(hex_polygon_bng(c).distance(Point(*_TO_BNG.transform(LON, LAT))) <= 3000 for c in big)


def test_hexes_within_radius_rejects_a_negative_radius():
    with pytest.raises(ValueError):
        hexes_within_radius(LAT, LON, -1)


# --- cell IDs typed by people ----------------------------------------------------------------


def test_is_valid_hex():
    assert is_valid_hex(HOME)
    for bad in ["", "hello", "88194ad147", "8", "zzzzzzzzzzzzzzz", None, 5]:
        assert not is_valid_hex(bad)


def test_a_valid_cell_of_the_wrong_resolution_is_refused():
    import h3

    if h3.__name__ == "fake_h3":
        pytest.skip("the stand-in only knows resolution 8")
    assert not is_valid_hex(h3.latlng_to_cell(LAT, LON, 9))


def test_parse_cells_accepts_commas_spaces_new_lines_capitals_and_blanks():
    other = neighbours(HOME)[0]
    text = f" {HOME.upper()},\n{other} ,, ;"
    assert parse_cells(text) == frozenset({HOME, other})


def test_parse_cells_of_nothing_is_empty():
    assert parse_cells("") == parse_cells("  ,, ") == parse_cells(None) == frozenset()


def test_parse_cells_accepts_a_list_and_removes_repeats():
    assert parse_cells([HOME, HOME]) == frozenset({HOME})


def test_parse_cells_names_the_bad_pieces_in_plain_english():
    with pytest.raises(InvalidCellError) as caught:
        parse_cells(f"{HOME}, banana, 12345")
    message = str(caught.value)
    assert '"banana"' in message and '"12345"' in message and HOME not in message
    assert "88194ad147fffff" in message  # shows what a good one looks like


def test_parse_cells_message_is_short_when_there_are_many_bad_pieces():
    with pytest.raises(InvalidCellError) as caught:
        parse_cells("a, b, c, d, e")
    assert "and more" in str(caught.value) and '"e"' not in str(caught.value)


def test_invalid_cell_error_is_a_value_error():
    assert issubclass(InvalidCellError, ValueError)
