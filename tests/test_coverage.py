import math

import pytest
from shapely.geometry import Point

from land_grab.coverage import (
    DEFAULT_MIN_DEPTH_M,
    MAX_SEGMENT_M,
    _densify_bng,
    count_unique_hexes,
    hexes_for_track,
    risky_hexes,
)
from land_grab.hexgrid import (
    _TO_BNG,
    _TO_WGS84,
    bng_to_hex,
    depth_in_hex,
    hex_centre_bng,
    hex_polygon_bng,
    latlon_to_hex,
    neighbours,
)

HEX = latlon_to_hex(51.5010, -0.1416)
_CX, _CY = hex_centre_bng(HEX)


def _latlon_from_bng(e, n):
    lon, lat = _TO_WGS84.transform(e, n)
    return lat, lon


def _track(*bng_points):
    return [_latlon_from_bng(e, n) for e, n in bng_points]


def _line_from_centre(de, dn, hex_id=HEX):
    """Two-point line starting at a cell centre, going de metres east and dn north."""
    e, n = hex_centre_bng(hex_id)
    return _track((e, n), (e + de, n + dn))


def _truth(start, de, dn):
    """Every cell touched by the line, found by sampling every 0.1 m."""
    return {bng_to_hex(start[0] + de * t / 20000, start[1] + dn * t / 20000) for t in range(20001)}


def test_straight_2km_line_crosses_a_handful_of_cells_starting_with_its_own():
    line = _line_from_centre(2000, 0)
    cells = hexes_for_track(line, min_depth_m=0)
    assert HEX in cells
    assert 2 <= len(cells) <= 5
    assert count_unique_hexes(line, min_depth_m=0) == len(cells)


def test_2km_line_matches_fine_sampling_at_any_start_and_angle():
    # Ground truth: sample every 0.1 m. Densifying at 20 m may only miss a cell
    # whose corner the line clips by less than 20 m, so allow at most one miss
    # and never an extra cell.
    e0, n0 = _TO_BNG.transform(-0.1416, 51.5010)
    for offset in range(0, 900, 150):
        for angle in range(0, 180, 30):
            de = 2000 * math.cos(math.radians(angle))
            dn = 2000 * math.sin(math.radians(angle))
            start = (e0 + offset, n0)
            line = _track(start, (start[0] + de, start[1] + dn))
            truth = _truth(start, de, dn)
            got = hexes_for_track(line, min_depth_m=0)
            assert got <= truth
            assert len(truth - got) <= 1
            assert 2 <= len(got) <= 8


def test_out_and_back_does_not_double_count():
    out = _line_from_centre(2000, 0)
    out_and_back = [out[0], out[1], out[0]]
    assert count_unique_hexes(out_and_back) == count_unique_hexes(out)


def test_closed_loop_does_not_double_count():
    corners = [(0, 0), (1500, 0), (1500, 1500), (0, 1500), (0, 0)]
    loop = _track(*[(_CX + de, _CY + dn) for de, dn in corners])
    cells = hexes_for_track(loop)
    # Repeating the whole loop a second time adds nothing.
    assert hexes_for_track(loop + loop) == cells
    assert count_unique_hexes(loop + loop) == len(cells)


def test_no_cell_is_skipped_on_a_long_segment():
    # Two points 5 km apart: without densifying only the 2 end cells would be seen.
    start = (_CX, _CY)
    line = _track(start, (_CX + 5000, _CY))
    cells = hexes_for_track(line, min_depth_m=0)
    assert len(cells) >= 5
    assert cells <= _truth(start, 5000, 0)
    assert len(_truth(start, 5000, 0) - cells) <= 1


def test_densify_keeps_segments_within_limit_and_keeps_original_points():
    line = _line_from_centre(1234, 567)
    dense = _densify_bng(line)
    for (x0, y0), (x1, y1) in zip(dense, dense[1:]):
        assert math.hypot(x1 - x0, y1 - y0) <= MAX_SEGMENT_M + 1e-6
    start = _TO_BNG.transform(line[0][1], line[0][0])
    end = _TO_BNG.transform(line[1][1], line[1][0])
    assert dense[0] == pytest.approx(start)
    assert dense[-1] == pytest.approx(end)


def test_empty_and_single_point_tracks():
    assert hexes_for_track([]) == set()
    assert count_unique_hexes([]) == 0
    point = _latlon_from_bng(_CX, _CY)
    assert hexes_for_track([point]) == {HEX}


def test_duplicate_consecutive_points_are_fine():
    point = _latlon_from_bng(_CX, _CY)
    assert count_unique_hexes([point, point, point]) == 1


# --- GPS safety margin (min_depth_m) -----------------------------------------------


def _corner_pass(depth_m):
    """A short pass that stays inside HEX, close to one of its corners and about ``depth_m`` deep."""
    polygon = hex_polygon_bng(HEX)
    (vx, vy) = polygon.exterior.coords[0]
    # Move from the corner towards the centre until the point is ``depth_m`` from the nearest edge.
    ux, uy = _CX - vx, _CY - vy
    length = math.hypot(ux, uy)
    ux, uy = ux / length, uy / length
    t = next(t / 10 for t in range(0, int(length) * 10) if depth_in_hex(vx + ux * t / 10, vy + uy * t / 10, HEX) >= depth_m)
    x, y = vx + ux * t, vy + uy * t
    # A pass of 10 m perpendicular to the direction of the centre (so it stays near that depth).
    return _track((x - uy * 5, y + ux * 5), (x + uy * 5, y - ux * 5))


def test_default_margin_is_25_m():
    assert DEFAULT_MIN_DEPTH_M == 25


def test_route_through_the_middle_of_a_cell_counts_it():
    track = _track((_CX - 100, _CY), (_CX + 100, _CY))
    assert hexes_for_track(track, 25) == {HEX}
    assert hexes_for_track(track) == {HEX}  # default margin
    assert risky_hexes(track, 25) == set()


def test_route_clipping_a_corner_only_counts_with_no_margin():
    track = _corner_pass(8)
    assert latlon_to_hex(*track[0]) == HEX

    assert HEX not in hexes_for_track(track, min_depth_m=25)
    assert HEX in hexes_for_track(track, min_depth_m=0)
    assert HEX in risky_hexes(track, min_depth_m=25)
    assert risky_hexes(track, min_depth_m=0) == set()


def test_a_pass_well_inside_a_corner_region_counts():
    track = _corner_pass(60)
    assert HEX in hexes_for_track(track, min_depth_m=25)


def test_margin_is_measured_to_the_real_cell_boundary():
    # One point a known distance inside the cell's real boundary (found with depth_in_hex).
    polygon = hex_polygon_bng(HEX)
    (ax, ay), (bx, by) = polygon.exterior.coords[0], polygon.exterior.coords[1]
    mid = ((ax + bx) / 2, (ay + by) / 2)
    length = math.hypot(mid[0] - _CX, mid[1] - _CY)
    for depth, counts in [(15, False), (40, True)]:
        f = (length - depth) / length
        point = _latlon_from_bng(_CX + (mid[0] - _CX) * f, _CY + (mid[1] - _CY) * f)
        assert (HEX in hexes_for_track([point], min_depth_m=25)) is counts


def test_margin_only_ever_removes_cells_and_risky_is_the_difference():
    e0, n0 = _TO_BNG.transform(-0.1416, 51.5010)
    line = _track((e0, n0), (e0 + 2000, n0 + 1300))
    everything = hexes_for_track(line, 0)
    safe = hexes_for_track(line, 25)
    risky = risky_hexes(line, 25)
    assert safe <= everything
    assert risky == everything - safe
    assert count_unique_hexes(line, 25) == len(safe)


def test_route_along_a_shared_edge_counts_neither_cell():
    # Run 5 m inside HEX along one of its edges: the neighbour across it is also only 5 m away.
    polygon = hex_polygon_bng(HEX)
    (ax, ay), (bx, by) = polygon.exterior.coords[0], polygon.exterior.coords[1]
    mid = ((ax + bx) / 2, (ay + by) / 2)
    length = math.hypot(mid[0] - _CX, mid[1] - _CY)
    f = (length - 5) / length
    px, py = _CX + (mid[0] - _CX) * f, _CY + (mid[1] - _CY) * f
    tx, ty = bx - ax, by - ay
    size = math.hypot(tx, ty)
    track = _track((px - tx / size * 100, py - ty / size * 100), (px + tx / size * 100, py + ty / size * 100))
    assert hexes_for_track(track, 25) == set()
    assert len(hexes_for_track(track, 0)) >= 1


def test_negative_margin_raises_value_error():
    with pytest.raises(ValueError):
        hexes_for_track([(51.5, -0.14)], -1)
    with pytest.raises(ValueError):
        risky_hexes([(51.5, -0.14)], -1)
    with pytest.raises(ValueError):
        count_unique_hexes([(51.5, -0.14)], -1)
