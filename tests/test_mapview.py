import re

import networkx as nx
import pytest
from shapely.geometry import LineString

from land_grab.hexgrid import hex_boundary, hex_centre, latlon_to_hex, neighbours
from land_grab.mapview import (
    CLAIMED_COLOUR,
    CLAIMED_TOOLTIP,
    COVERED_COLOUR,
    ROUTE_COLOUR,
    draw_map,
    route_to_latlons,
    surrounding_hexes,
)

HOME = latlon_to_hex(51.5, -0.12)
_NEXT, _AFTER, _FAR = neighbours(HOME)[0], neighbours(neighbours(HOME)[0])[0], neighbours(HOME)[3]

START = hex_centre(HOME)
ROUTE = [START, hex_centre(_NEXT), hex_centre(_AFTER)]
COVERED = {HOME, _NEXT, _AFTER}


def _polygon_fills(html):
    """The "fill" flag of every L.polygon in the map (the route line is an L.polyline)."""
    fills = []
    for match in re.finditer(r"L\.polygon\(", html):
        flag = re.search(r'"fill":\s*(true|false)', html[match.start() :])
        fills.append(flag.group(1) == "true")
    return fills


def _html(tmp_path, **kwargs):
    out = tmp_path / "map.html"
    returned = draw_map(START, ROUTE, COVERED, out, **kwargs)
    assert returned == out
    return out.read_text()


def test_saves_a_folium_html_file(tmp_path):
    html = _html(tmp_path)
    assert "<html" in html
    assert "leaflet" in html.lower()


def test_creates_missing_parent_folders(tmp_path):
    out = tmp_path / "a" / "b" / "map.html"
    draw_map(START, ROUTE, COVERED, out)
    assert out.exists()


def test_covered_hexagons_are_filled_and_the_rest_are_not(tmp_path):
    html = _html(tmp_path, margin_rings=1)

    fills = _polygon_fills(html)
    filled, outlined = fills.count(True), fills.count(False)
    expected_outline = len(surrounding_hexes(COVERED, 1) - COVERED)

    assert filled == len(COVERED)
    assert outlined == expected_outline > 0
    assert COVERED_COLOUR in html


def test_route_is_drawn_as_a_line_through_all_points(tmp_path):
    html = _html(tmp_path)
    assert ROUTE_COLOUR in html
    for lat, lon in ROUTE:
        assert str(lat) in html


def test_start_point_is_marked(tmp_path):
    html = _html(tmp_path)
    assert "L.marker" in html
    assert "Start" in html


def test_empty_route_still_makes_a_map(tmp_path):
    out = tmp_path / "map.html"
    draw_map(START, [], {HOME}, out)
    assert ROUTE_COLOUR not in out.read_text()


def test_margin_rings_zero_draws_no_outline_hexagons(tmp_path):
    html = _html(tmp_path, margin_rings=0)
    assert _polygon_fills(html) == [True] * len(COVERED)


def test_invalid_arguments_raise_value_error(tmp_path):
    with pytest.raises(ValueError):
        draw_map(START, ROUTE, COVERED, tmp_path / "m.html", margin_rings=-1)


# --- helpers -------------------------------------------------------------------


def test_surrounding_hexes_ring_sizes():
    assert surrounding_hexes({HOME}, 0) == {HOME}
    assert len(surrounding_hexes({HOME}, 1)) == 7
    assert len(surrounding_hexes({HOME}, 2)) == 19
    assert HOME in surrounding_hexes({HOME}, 2)


def test_route_to_latlons_follows_geometry_and_orients_it():
    g = nx.MultiDiGraph()
    g.add_node("a", y=51.0, x=-0.10)
    g.add_node("b", y=51.0, x=-0.08)
    g.add_node("c", y=51.01, x=-0.08)
    # a -> b has a bend stored in the opposite direction (b to a)
    bend = LineString([(-0.08, 51.0), (-0.09, 51.005), (-0.10, 51.0)])
    g.add_edge("a", "b", length=1, geometry=bend)
    g.add_edge("b", "c", length=1)  # no geometry: straight to c

    assert route_to_latlons(g, ["a", "b", "c"]) == [
        (51.0, -0.10),
        (51.005, -0.09),
        (51.0, -0.08),
        (51.01, -0.08),
    ]


def test_route_to_latlons_empty_and_single_node():
    g = nx.MultiDiGraph()
    g.add_node("a", y=51.0, x=-0.10)
    assert route_to_latlons(g, []) == []
    assert route_to_latlons(g, ["a"]) == [(51.0, -0.10)]


# --- risky hexagons --------------------------------------------------------------------

from land_grab.mapview import RISKY_COLOUR, RISKY_TOOLTIP  # noqa: E402

RISKY = {_FAR}


def _polygon_blocks(html):
    """Each L.polygon(...) call's text, up to the next polygon/polyline/marker."""
    starts = [m.start() for m in re.finditer(r"L\.polygon\(", html)]
    ends = starts[1:] + [len(html)]
    return [html[a:b] for a, b in zip(starts, ends)]


def test_risky_hexagons_are_drawn_lighter_with_a_dashed_outline(tmp_path):
    html = _html(tmp_path, margin_rings=0, risky_hex_ids=RISKY)

    risky_blocks = [b for b in _polygon_blocks(html) if RISKY_COLOUR in b]
    assert len(risky_blocks) == len(RISKY)
    block = risky_blocks[0]
    assert re.search(r'"dashArray":\s*"6 6"', block)
    assert '"fill": true' in block
    assert RISKY_TOOLTIP in block
    assert RISKY_TOOLTIP == "risky: route only clips this hexagon"

    # The safe hexagons keep their solid outline.
    safe_blocks = [b for b in _polygon_blocks(html) if COVERED_COLOUR in b]
    assert len(safe_blocks) == len(COVERED)
    assert all("dashArray" not in b or '"dashArray": null' in b for b in safe_blocks)
    assert all(RISKY_TOOLTIP not in b for b in safe_blocks)


def test_no_risky_hexagons_by_default(tmp_path):
    html = _html(tmp_path)
    assert RISKY_COLOUR not in html
    assert RISKY_TOOLTIP not in html


def test_a_hexagon_both_safe_and_risky_is_drawn_as_safe(tmp_path):
    html = _html(tmp_path, margin_rings=0, risky_hex_ids={HOME})
    assert RISKY_COLOUR not in html
    assert _polygon_fills(html) == [True] * len(COVERED)


def test_risky_hexagons_are_not_also_drawn_as_faint_outlines(tmp_path):
    html = _html(tmp_path, margin_rings=1, risky_hex_ids=RISKY)
    expected_outline = len(surrounding_hexes(COVERED | RISKY, 1) - COVERED - RISKY)
    fills = _polygon_fills(html)
    assert fills.count(True) == len(COVERED) + len(RISKY)
    assert fills.count(False) == expected_outline


# --- the hexagons are the real H3 cells -------------------------------------------------


def test_each_hexagon_is_drawn_from_its_h3_boundary(tmp_path):
    html = _html(tmp_path, margin_rings=0)
    for hex_id in COVERED:
        for lat, lon in hex_boundary(hex_id):
            assert str(lat) in html
            assert str(lon) in html


def test_hovering_a_covered_hexagon_shows_its_cell_id(tmp_path):
    html = _html(tmp_path, margin_rings=0)
    for hex_id in COVERED:
        assert hex_id in html


# --- cells you already own --------------------------------------------------------------

CLAIMED = {_FAR}


def test_claimed_cells_are_drawn_in_their_own_colour_with_a_tooltip(tmp_path):
    html = _html(tmp_path, margin_rings=0, claimed_hex_ids=CLAIMED)

    blocks = [b for b in _polygon_blocks(html) if CLAIMED_COLOUR in b]
    assert len(blocks) == len(CLAIMED)
    assert f"{_FAR} - {CLAIMED_TOOLTIP}" in blocks[0]
    assert CLAIMED_COLOUR != COVERED_COLOUR != RISKY_COLOUR


def test_no_claimed_colour_when_nothing_is_claimed(tmp_path):
    html = _html(tmp_path)
    assert CLAIMED_COLOUR not in html and CLAIMED_TOOLTIP not in html


def test_a_claimed_cell_is_never_also_drawn_as_covered_or_risky(tmp_path):
    html = _html(tmp_path, margin_rings=0, claimed_hex_ids={HOME, _FAR}, risky_hex_ids={_FAR})
    blocks = _polygon_blocks(html)
    assert len([b for b in blocks if CLAIMED_COLOUR in b]) == 2
    assert len([b for b in blocks if COVERED_COLOUR in b]) == len(COVERED) - 1
    assert RISKY_COLOUR not in html
