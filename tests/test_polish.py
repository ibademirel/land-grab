import pytest

from land_grab.optimiser import optimise_route, score_route
from land_grab.polish import polish_route
from land_grab.router import route_length
from test_optimiser import HOME_CELL, START, TARGET_M, _grid, _rectangle, _walk


@pytest.fixture
def graph():
    return _grid()


def _with_detour():
    """The 15 x 5 rectangle with a pointless notch (up 2, across 2, down 2) on its bottom edge."""
    x, y = START
    return _walk(
        START, (x + 4, y), (x + 4, y + 2), (x + 6, y + 2), (x + 6, y),
        (x + 15, y), (x + 15, y + 5), (x, y + 5), START,
    )  # fmt: skip


def _within(length, target, tolerance):
    return abs(length - target) <= target * tolerance + 1e-6


# --- step 1: detours ---------------------------------------------------------------


def test_pointless_detour_is_removed_without_losing_hexagons(graph):
    detour = _with_detour()
    assert route_length(graph, detour) == 4400

    result = polish_route(graph, detour, target_m=4000, tolerance=0.0)

    assert result.route == _rectangle(15, 5)
    assert result.length_before_m == 4400
    assert result.length_after_m == 4000
    assert result.hexes_after >= result.hexes_before == 5


def test_detour_is_not_cut_if_that_would_lose_a_hexagon(graph):
    # A long excursion into hexagons that the rest of the route never enters.
    x, y = START
    excursion = _walk(START, (x + 10, y), (x + 10, y - 9), (x + 10, y), (x, y), (x, y + 2), START)
    before = score_route(graph, excursion).unique_hexes

    result = polish_route(graph, excursion, target_m=route_length(graph, excursion), tolerance=0.0)

    assert result.hexes_after >= before
    assert score_route(graph, result.route).unique_hexes >= before


# --- step 2: spare distance --------------------------------------------------------


def test_spare_distance_picks_up_one_extra_hexagon(graph):
    route = _rectangle(14, 4)  # 3600 m, 5 hexagons: 400 m spare against a 4000 m target
    assert route_length(graph, route) == 3600
    assert score_route(graph, route).unique_hexes == 5

    result = polish_route(graph, route, target_m=4000, tolerance=0.05)

    assert result.hexes_before == 5
    assert result.hexes_after == 6
    assert result.length_after_m > result.length_before_m
    assert _within(result.length_after_m, 4000, 0.05)
    assert score_route(graph, result.route).unique_hexes == 6


def test_extra_hexagons_are_never_bought_outside_the_tolerance(graph):
    route = _rectangle(14, 4)
    # 3600 m against a 3600 m +/- 1% target: no room for any detour into a new hexagon.
    result = polish_route(graph, route, target_m=3600, tolerance=0.01)
    assert result.route == route
    assert _within(result.length_after_m, 3600, 0.01)


# --- invariants and summary -----------------------------------------------------------


@pytest.mark.parametrize("route", [_with_detour(), _rectangle(14, 4), _rectangle(15, 5)])
def test_result_is_closed_and_within_tolerance(graph, route):
    result = polish_route(graph, route, target_m=TARGET_M, tolerance=0.05)

    assert result.route[0] == result.route[-1] == START
    assert all(graph.has_edge(u, v) for u, v in zip(result.route, result.route[1:]))
    assert _within(result.length_after_m, TARGET_M, 0.05)
    assert result.length_after_m == route_length(graph, result.route)
    assert result.hexes_after >= result.hexes_before


def test_route_that_cannot_be_improved_is_returned_unchanged(graph):
    route = _rectangle(12, 4)  # 3200 m, 4 hexagons, no detours, no spare distance
    result = polish_route(graph, route, target_m=3200, tolerance=0.0)

    assert result.route == route
    assert result.rounds == 0
    assert result.hexes_after == result.hexes_before
    assert result.length_after_m == result.length_before_m
    assert result.wasted_after_m == result.wasted_before_m


def test_summary_reports_before_and_after(graph):
    detour = _with_detour()
    before = score_route(graph, detour)

    result = polish_route(graph, detour, target_m=4000, tolerance=0.0)
    after = score_route(graph, result.route)

    assert result.hexes_before == before.unique_hexes
    assert result.wasted_before_m == pytest.approx(before.wasted_m)
    assert result.hexes_after == after.unique_hexes
    assert result.wasted_after_m == pytest.approx(after.wasted_m)
    assert result.length_before_m == route_length(graph, detour)
    assert result.rounds >= 1


def test_max_rounds_zero_changes_nothing(graph):
    detour = _with_detour()
    result = polish_route(graph, detour, target_m=4000, tolerance=0.0, max_rounds=0)
    assert result.route == detour


def test_length_ignores_the_street_reuse_penalty(graph):
    # Walking the same street twice is penalised when routing, never in the reported length.
    out_and_back = _walk(START, (START[0] + 20, START[1]), START)
    result = polish_route(graph, out_and_back, target_m=4000, tolerance=0.05)
    assert result.length_after_m == route_length(graph, result.route)


def test_polishing_an_optimised_route_never_makes_it_worse(graph):
    best = optimise_route(graph, START, TARGET_M, n_candidates=20, verbose=False)
    result = polish_route(graph, best.route, TARGET_M)

    assert result.hexes_after >= best.unique_hexes
    assert result.route[0] == result.route[-1] == START
    assert _within(result.length_after_m, TARGET_M, 0.05)


def test_invalid_arguments_raise_value_error(graph):
    route = _rectangle(15, 5)
    with pytest.raises(ValueError):
        polish_route(graph, [], 4000)
    with pytest.raises(ValueError):
        polish_route(graph, route[:-1], 4000)  # not closed
    with pytest.raises(ValueError):
        polish_route(graph, route, 0, 4000)
    with pytest.raises(ValueError):
        polish_route(graph, route, 0)
    with pytest.raises(ValueError):
        polish_route(graph, route, 4000, tolerance=-0.1)
    with pytest.raises(ValueError):
        polish_route(graph, route, 4000, max_rounds=-1)


# --- spurs and zigzags ------------------------------------------------------------------

import networkx as nx  # noqa: E402

from land_grab.hexgrid import _TO_WGS84, hex_centre_bng  # noqa: E402
from land_grab.polish import find_route_quirks  # noqa: E402


def _with_spur(route, column, depth, direction=-1):
    """Insert an out-and-back stub of ``depth`` blocks off the route's node (START.x + column, START.y)."""
    x, y = START
    i = route.index((x + column, y))
    out = [(x + column, y + direction * k) for k in range(1, depth + 1)]
    return route[: i + 1] + out + out[-2::-1] + [(x + column, y)] + route[i + 1 :]


def _small_graph(coords, edges, one_way=()):
    """Tiny graph with nodes at metre offsets (east, north) from a hexagon centre."""
    cx, cy = hex_centre_bng(HOME_CELL)
    g = nx.MultiDiGraph()
    for name, (de, dn) in coords.items():
        lon, lat = _TO_WGS84.transform(cx + de, cy + dn)
        g.add_node(name, x=lon, y=lat)
    for a, b in edges:
        length = ((coords[a][0] - coords[b][0]) ** 2 + (coords[a][1] - coords[b][1]) ** 2) ** 0.5
        g.add_edge(a, b, length=length)
        if (a, b) not in one_way:
            g.add_edge(b, a, length=length)
    return g


def test_find_route_quirks_reports_a_spur_with_length_and_position(graph):
    route = _with_spur(_rectangle(15, 5), column=7, depth=1)  # 100 m up the street and back
    (spur,) = find_route_quirks(graph, route, max_length_m=250)

    assert spur.kind == "spur"
    assert spur.length_m == spur.saving_m == 200
    assert route[spur.start] == route[spur.end] == (START[0] + 7, START[1])
    tip_lat, tip_lon = graph.nodes[(START[0] + 7, START[1] - 1)]["y"], graph.nodes[(START[0] + 7, START[1] - 1)]["x"]
    assert spur.latlon == (tip_lat, tip_lon)


def test_find_route_quirks_ignores_spurs_longer_than_the_limit_and_whole_out_and_backs(graph):
    long_spur = _with_spur(_rectangle(15, 5), column=7, depth=3)  # 600 m walked
    assert find_route_quirks(graph, long_spur, max_length_m=150) == []
    assert find_route_quirks(graph, long_spur, max_length_m=700)[0].length_m == 600

    out_and_back = _walk(START, (START[0] + 1, START[1]), START)
    assert find_route_quirks(graph, out_and_back, max_length_m=1000) == []  # the route is the spur


def test_find_route_quirks_reports_a_zigzag_with_a_shorter_legal_path():
    coords = {"a": (0, 0), "b": (0, 50), "c": (50, 50), "d": (50, 120), "e": (-60, 120)}
    g = _small_graph(coords, [("a", "b"), ("b", "c"), ("a", "c"), ("c", "d"), ("d", "e"), ("e", "a")])
    route = ["a", "b", "c", "d", "e", "a"]

    (zigzag,) = find_route_quirks(g, route)

    assert zigzag.kind == "zigzag"
    assert (route[zigzag.start], route[zigzag.end]) == ("a", "c")
    assert zigzag.length_m == pytest.approx(100)
    assert zigzag.saving_m == pytest.approx(100 - 50 * 2**0.5)
    assert zigzag.shortcut == ("a", "c")
    assert zigzag.latlon == (g.nodes["b"]["y"], g.nodes["b"]["x"])


def test_a_zigzag_with_no_legal_shorter_path_is_not_a_quirk():
    # A real road crossing: the diagonal exists only in the other direction (one-way),
    # or not at all, so there is nothing shorter that can be walked from a to c.
    coords = {"a": (0, 0), "b": (0, 50), "c": (50, 50), "d": (50, 120), "e": (-60, 120)}
    ring = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "a")]
    route = ["a", "b", "c", "d", "e", "a"]

    assert find_route_quirks(_small_graph(coords, ring), route) == []
    one_way = _small_graph(coords, ring + [("c", "a")], one_way=[("c", "a")])
    assert find_route_quirks(one_way, route) == []


def test_zigzag_is_removed_when_a_shorter_legal_path_exists():
    coords = {"a": (0, 0), "b": (0, 50), "c": (50, 50), "d": (50, 120), "e": (-60, 120)}
    g = _small_graph(coords, [("a", "b"), ("b", "c"), ("a", "c"), ("c", "d"), ("d", "e"), ("e", "a")])
    route = ["a", "b", "c", "d", "e", "a"]

    result = polish_route(g, route, target_m=route_length(g, route), tolerance=0.2)

    assert result.route == ["a", "c", "d", "e", "a"]
    assert result.length_after_m < result.length_before_m
    assert result.hexes_after >= result.hexes_before
    assert len(result.quirks_before) == 1
    assert result.quirks_kept == []


def test_pointless_spur_is_removed_and_hexagons_are_kept(graph):
    base = _rectangle(15, 5)
    route = _with_spur(base, column=7, depth=1)
    assert route_length(graph, route) == 4200
    before = score_route(graph, route)

    # Target 4000 m exactly: the spur is the only thing standing between the route and the target.
    result = polish_route(graph, route, target_m=4000, tolerance=0.0, quirk_max_m=250)

    assert result.route == base
    assert result.length_after_m == 4000
    assert result.hexes_after == result.hexes_before == before.unique_hexes
    assert [q.kind for q in result.quirks_before] == ["spur"]
    assert result.quirks_kept == []


def test_a_spur_that_is_the_only_way_into_a_safe_hexagon_is_kept(graph):
    base = _walk(START, (START[0] + 6, START[1]), (START[0] + 6, START[1] + 2), (START[0], START[1] + 2), START)
    route = _with_spur(base, column=3, depth=4)  # 800 m stub into a hexagon the loop never enters
    assert score_route(graph, route).unique_hexes == score_route(graph, base).unique_hexes + 1

    result = polish_route(
        graph, route, target_m=route_length(graph, route), tolerance=0.0, quirk_max_m=1000
    )

    assert result.route == route
    assert result.hexes_after == result.hexes_before
    spur = next(k for k in result.quirks_kept if k.quirk.kind == "spur")
    assert spur.reason == "earns a safe hexagon"
    assert len(spur.hexes) == 1


def test_a_spur_that_only_earns_a_risky_hexagon_is_removed():
    g = _grid(offset_e=-1880)
    base = _walk(START, (START[0] + 6, START[1]), (START[0] + 6, START[1] + 2), (START[0], START[1] + 2), START)
    route = _with_spur(base, column=4, depth=3)
    length = route_length(g, base)
    assert score_route(g, route, min_depth_m=0).unique_hexes == 3
    assert score_route(g, route, min_depth_m=25).unique_hexes == 2

    margin = polish_route(g, route, target_m=length, tolerance=0.0, quirk_max_m=700)
    no_margin = polish_route(g, route, target_m=length, tolerance=0.0, quirk_max_m=700, min_depth_m=0)

    assert margin.route == base  # the spur only touches a hexagon without properly entering it
    assert no_margin.hexes_after == 3  # with no margin the spur does earn a hexagon, so it stays
    assert any(k.reason == "earns a safe hexagon" for k in no_margin.quirks_kept)


def test_a_spur_is_kept_if_removing_it_would_leave_the_length_band(graph):
    route = _with_spur(_rectangle(15, 5), column=7, depth=1)  # 4200 m
    result = polish_route(graph, route, target_m=4200, tolerance=0.0, quirk_max_m=250)
    assert result.route == route
    assert [k.reason for k in result.quirks_kept] == ["needed to stay within the length limit"]


def test_route_without_quirks_is_unchanged(graph):
    route = _rectangle(12, 4)
    assert find_route_quirks(graph, route) == []

    result = polish_route(graph, route, target_m=3200, tolerance=0.0)

    assert result.route == route
    assert result.quirks_before == [] and result.quirks_kept == []


def test_removing_quirks_never_loses_a_safe_hexagon(graph):
    route = _with_spur(_with_spur(_rectangle(15, 5), 3, 1), 11, 1, direction=1)
    result = polish_route(graph, route, target_m=4000, tolerance=0.1)
    assert result.hexes_after >= result.hexes_before
    assert result.route[0] == result.route[-1] == START


def test_negative_quirk_max_raises_value_error(graph):
    with pytest.raises(ValueError):
        polish_route(graph, _rectangle(15, 5), 4000, quirk_max_m=-1)
