import math
import random

import networkx as nx
import pytest

from land_grab import optimiser
from land_grab.hexgrid import _TO_WGS84, depth_in_hex, hex_centre_bng, latlon_to_hex, neighbours
from land_grab.optimiser import optimise_route, score_route
from land_grab.router import LoopNotFoundError, route_length

BLOCK_M = 100
TARGET_M = 4000
START = (20, 10)

# The grid is centred on the H3 cell holding central London: 41 x 21 nodes, 100 m apart,
# so a 4 km route spans a handful of H3 cells (each about 0.9 km across).
HOME_CELL = latlon_to_hex(51.5, -0.12)
# The grid is moved 200 m south of the cell's centre so that, on the real H3 grid, a 15 x 5 block loop
# crosses five cells and an out-and-back crosses three (the tests below rely on those counts).
_E0, _N0 = hex_centre_bng(HOME_CELL)
_N0 -= 200


def _grid(nx_=41, ny=21, offset_e=-2000, offset_n=-1000):
    g = nx.MultiDiGraph()
    for i in range(nx_):
        for j in range(ny):
            lon, lat = _TO_WGS84.transform(
                _E0 + offset_e + i * BLOCK_M, _N0 + offset_n + j * BLOCK_M
            )
            g.add_node((i, j), x=lon, y=lat)
    for u, v in nx.grid_2d_graph(nx_, ny).edges():
        g.add_edge(u, v, length=BLOCK_M)
        g.add_edge(v, u, length=BLOCK_M)
    return g


def _walk(*corners):
    """Node-by-node route through the corner nodes (horizontal, then vertical moves)."""
    route = [corners[0]]
    for cx, cy in corners[1:]:
        x, y = route[-1]
        while x != cx:
            x += 1 if cx > x else -1
            route.append((x, y))
        while y != cy:
            y += 1 if cy > y else -1
            route.append((x, y))
    return route


def _rectangle(width, height):
    x, y = START
    return _walk(START, (x + width, y), (x + width, y + height), (x, y + height), START)


def _out_and_back():
    return _walk(START, (START[0] + 20, START[1]), START)


@pytest.fixture
def graph():
    return _grid()


def _fake_make_loop(monkeypatch, routes):
    """Make make_loop return the given routes in turn (an exception instance is raised)."""
    queue = iter(routes)

    def fake(graph, start_node, target_m, seed=None, **kwargs):
        item = next(queue)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(optimiser, "make_loop", fake)


# --- scoring ---------------------------------------------------------------


def test_out_and_back_scores_lower_than_a_loop_of_the_same_length(graph):
    out_and_back = _out_and_back()
    loop = _rectangle(width=15, height=5)
    assert route_length(graph, out_and_back) == route_length(graph, loop) == TARGET_M

    back = score_route(graph, out_and_back)
    around = score_route(graph, loop)

    assert around.unique_hexes > back.unique_hexes
    assert around.wasted_m < back.wasted_m


def test_each_hexagon_counts_once(graph):
    # Going round the same loop twice claims no extra hexagons.
    loop = _rectangle(width=15, height=5)
    twice = loop + loop[1:]
    assert score_route(graph, twice).unique_hexes == score_route(graph, loop).unique_hexes


def test_wasted_distance_counts_only_hexagons_already_claimed(graph):
    # A straight line out into new hexagons wastes nothing.
    straight = _walk(START, (START[0] + 20, START[1]))
    assert score_route(graph, straight).wasted_m == pytest.approx(0)

    # Doing it twice (out, back, out again) wastes at least the whole return trip
    # outside the hexagon being left.
    there_and_back = _out_and_back()
    assert score_route(graph, there_and_back).wasted_m > 0.25 * route_length(
        graph, there_and_back
    )


def test_wasted_distance_never_exceeds_route_length(graph):
    route = _out_and_back()
    assert 0 <= score_route(graph, route).wasted_m <= route_length(graph, route)


def test_single_node_route_claims_one_hexagon(graph):
    assert score_route(graph, [START]) == optimiser.RouteScore(unique_hexes=1, wasted_m=0)


# --- optimiser, with controlled candidates ------------------------------------


def test_optimiser_picks_the_route_covering_the_most_hexagons(graph, monkeypatch):
    poor, good = _out_and_back(), _rectangle(width=15, height=5)
    assert score_route(graph, good).unique_hexes > score_route(graph, poor).unique_hexes

    for order in ([poor, good], [good, poor]):
        _fake_make_loop(monkeypatch, order)
        result = optimise_route(graph, START, TARGET_M, n_candidates=2, verbose=False)
        assert result.route == good
        assert result.unique_hexes == 5
        assert result.length_m == TARGET_M


def test_equal_scores_are_broken_by_less_wasted_distance(graph, monkeypatch):
    tidy = _rectangle(width=15, height=5)  # 5 hexagons, 600 m wasted
    untidy = _rectangle(width=16, height=4)  # 5 hexagons, 1000 m wasted
    a, b = score_route(graph, tidy), score_route(graph, untidy)
    assert a.unique_hexes == b.unique_hexes
    assert a.wasted_m < b.wasted_m

    for order in ([tidy, untidy], [untidy, tidy]):
        _fake_make_loop(monkeypatch, order)
        result = optimise_route(graph, START, TARGET_M, n_candidates=2, verbose=False)
        assert result.route == tidy
        assert result.wasted_m == pytest.approx(a.wasted_m)


def test_candidates_outside_five_percent_are_discarded(graph, monkeypatch):
    ok = _rectangle(width=15, height=5)  # 4000 m, 5 hexagons
    too_long = _rectangle(width=20, height=10)  # 6000 m, more hexagons but outside tolerance
    assert score_route(graph, too_long).unique_hexes > score_route(graph, ok).unique_hexes

    _fake_make_loop(monkeypatch, [too_long, ok])
    result = optimise_route(graph, START, TARGET_M, n_candidates=2, verbose=False)

    assert result.route == ok
    assert result.candidates_tried == 2
    assert result.candidates_kept == 1


def test_failed_seeds_are_skipped(graph, monkeypatch):
    good = _rectangle(width=15, height=5)
    _fake_make_loop(monkeypatch, [LoopNotFoundError("nope"), good])
    result = optimise_route(graph, START, TARGET_M, n_candidates=2, verbose=False)
    assert result.route == good
    assert result.candidates_kept == 1


def test_raises_if_no_candidate_is_kept(graph, monkeypatch):
    _fake_make_loop(monkeypatch, [_rectangle(width=20, height=10)] * 3)
    with pytest.raises(LoopNotFoundError, match="None of 3 candidates"):
        optimise_route(graph, START, TARGET_M, n_candidates=3, verbose=False)


def test_prints_a_short_summary(graph, monkeypatch, capsys):
    _fake_make_loop(monkeypatch, [_rectangle(width=15, height=5)])
    optimise_route(graph, START, TARGET_M, n_candidates=1)
    out = capsys.readouterr().out
    assert "candidates tried: 1" in out
    expected = score_route(graph, _rectangle(width=15, height=5))
    assert f"best score: {expected.unique_hexes} hexagons" in out
    assert "best length: 4.00 km" in out
    assert f"wasted distance: {expected.wasted_m / 1000:.2f} km" in out


def test_verbose_false_prints_nothing(graph, monkeypatch, capsys):
    _fake_make_loop(monkeypatch, [_rectangle(width=15, height=5)])
    optimise_route(graph, START, TARGET_M, n_candidates=1, verbose=False)
    assert capsys.readouterr().out == ""


# --- optimiser, with the real make_loop ---------------------------------------------


def test_real_candidates_best_score_is_the_maximum(graph):
    n, seed = 40, 0
    result = optimise_route(graph, START, TARGET_M, n_candidates=n, seed=seed, verbose=False)

    # Re-create the same candidates and score each one on its own.
    rng = random.Random(seed)
    scores = [
        score_route(graph, optimiser.make_loop(graph, START, TARGET_M, seed=rng.getrandbits(32)))
        for _ in range(n)
    ]
    best = max(s.unique_hexes for s in scores)
    assert result.unique_hexes == best
    assert result.wasted_m == min(s.wasted_m for s in scores if s.unique_hexes == best)
    assert abs(result.length_m - TARGET_M) <= 0.05 * TARGET_M
    assert result.route[0] == result.route[-1] == START


def test_same_seed_gives_same_result(graph):
    kwargs = dict(n_candidates=20, verbose=False)
    a = optimise_route(graph, START, TARGET_M, seed=3, **kwargs)
    b = optimise_route(graph, START, TARGET_M, seed=3, **kwargs)
    assert a.route == b.route


def test_invalid_n_candidates_raises_value_error(graph):
    with pytest.raises(ValueError):
        optimise_route(graph, START, TARGET_M, n_candidates=0)


# --- GPS safety margin ----------------------------------------------------------------

from land_grab.coverage import DEFAULT_MIN_DEPTH_M  # noqa: E402


STEP_M = 5


def _line_across_an_edge(depths_m=(10, 60)):
    """A straight line of nodes (5 m apart) from the home cell's centre across its edge into a neighbour.

    Returns (graph, neighbour, {depth: route}): each route runs along the line to the first node
    at least that many metres inside the neighbour, so a 10 m route only clips the neighbour.
    """
    other = neighbours(HOME_CELL)[0]
    (x0, y0), (x1, y1) = hex_centre_bng(HOME_CELL), hex_centre_bng(other)
    gap = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / gap, (y1 - y0) / gap
    g = nx.MultiDiGraph()
    depths = []
    for i in range(int(gap / STEP_M)):
        x, y = x0 + ux * i * STEP_M, y0 + uy * i * STEP_M
        lon, lat = _TO_WGS84.transform(x, y)
        g.add_node(i, x=lon, y=lat)
        depths.append(depth_in_hex(x, y, other))
        if i:
            g.add_edge(i - 1, i, length=STEP_M)
            g.add_edge(i, i - 1, length=STEP_M)
    routes = {want: list(range(next(i for i, d in enumerate(depths) if d >= want) + 1)) for want in depths_m}
    return g, other, routes


def test_score_route_default_margin_is_25_m():
    assert DEFAULT_MIN_DEPTH_M == 25


def test_route_that_only_clips_a_cell_scores_lower_with_the_margin():
    g, other, routes = _line_across_an_edge()
    clipping, deep = routes[10], routes[60]

    assert score_route(g, clipping, min_depth_m=0).unique_hexes == 2
    assert score_route(g, clipping, min_depth_m=25).unique_hexes == 1
    assert score_route(g, clipping).unique_hexes == 1  # default margin
    assert score_route(g, deep, min_depth_m=25).unique_hexes == 2


def test_zero_margin_scores_at_least_as_many_cells_and_wasted_distance_is_unchanged(graph):
    route = _rectangle(width=15, height=5)
    loose, strict = score_route(graph, route, min_depth_m=0), score_route(graph, route, min_depth_m=25)
    assert loose.unique_hexes >= strict.unique_hexes >= 1
    assert loose.wasted_m == pytest.approx(strict.wasted_m)


def test_optimiser_prefers_the_route_that_goes_well_inside_cells(monkeypatch):
    g, other, routes = _line_across_an_edge()
    clipping, deep = routes[10], routes[60]
    target = STEP_M * (len(deep) - 1)

    # With no margin the two routes count the same cells, and the shorter (clipping) route wastes less ...
    a, b = score_route(g, clipping, min_depth_m=0), score_route(g, deep, min_depth_m=0)
    assert a.unique_hexes == b.unique_hexes == 2

    # ... but with the margin on, only the deep route claims the second cell, whatever the order.
    for order in ([clipping, deep], [deep, clipping]):
        _fake_make_loop(monkeypatch, order)
        result = optimise_route(g, 0, target, n_candidates=2, tolerance=0.5, verbose=False)
        assert result.route == deep
        assert result.unique_hexes == 2


def test_negative_margin_raises_value_error(graph):
    with pytest.raises(ValueError):
        score_route(graph, _rectangle(15, 5), min_depth_m=-1)
    with pytest.raises(ValueError):
        optimise_route(graph, START, TARGET_M, n_candidates=1, min_depth_m=-1)


# --- cells you already own ---------------------------------------------------------------


def test_claimed_cells_are_not_counted_as_new(graph):
    route = _rectangle(width=15, height=5)
    everything = score_route(graph, route)
    from land_grab.coverage import hexes_for_track
    from land_grab.mapview import route_to_latlons

    cells = sorted(hexes_for_track(route_to_latlons(graph, route), 25))
    assert everything.unique_hexes == len(cells)

    one = score_route(graph, route, claimed=frozenset(cells[:1]))
    assert one.unique_hexes == len(cells) - 1
    assert score_route(graph, route, claimed=frozenset(cells)).unique_hexes == 0


def test_claimed_cells_do_not_change_the_wasted_distance(graph):
    route = _rectangle(width=15, height=5)
    from land_grab.coverage import hexes_for_track
    from land_grab.mapview import route_to_latlons

    cells = frozenset(hexes_for_track(route_to_latlons(graph, route), 25))
    assert score_route(graph, route, claimed=cells).wasted_m == pytest.approx(score_route(graph, route).wasted_m)


def test_the_optimiser_prefers_the_route_with_more_new_cells(monkeypatch):
    g = _grid()
    from land_grab.coverage import hexes_for_track
    from land_grab.mapview import route_to_latlons

    west = _walk(START, (START[0] - 15, START[1]), START)
    east = _walk(START, (START[0] + 15, START[1]), START)
    cells = {name: hexes_for_track(route_to_latlons(g, r), 25) for name, r in (("west", west), ("east", east))}
    assert cells["east"] - cells["west"] and cells["west"] - cells["east"]  # each reaches a cell the other does not

    # If you already own what the east route would reach, the west route is the better one.
    for order in ([west, east], [east, west]):
        _fake_make_loop(monkeypatch, order)
        result = optimise_route(
            g, START, 3000, n_candidates=2, claimed=frozenset(cells["east"] - cells["west"]), verbose=False
        )
        assert result.route == west
