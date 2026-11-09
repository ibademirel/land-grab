import networkx as nx
import pytest

from land_grab import planner
from land_grab.coverage import hexes_for_track
from land_grab.hexgrid import (
    _TO_BNG,
    _TO_WGS84,
    bng_to_hex,
    depth_in_hex,
    hex_centre_bng,
    hex_distance,
    hexes_within_radius,
    neighbours,
)
from land_grab.mapview import route_to_latlons
from land_grab.optimiser import optimise_route, score_route
from land_grab.planner import (
    SKIPPED,
    STAGE_CHOOSING,
    STAGE_FALLBACK,
    STAGE_JOINING,
    PlannerError,
    PlannerTimeout,
    SearchResult,
    choose_route,
    choose_route_fast,
    find_anchors,
    format_choice,
    plan_route,
)
from land_grab.router import LoopNotFoundError, route_length
from test_optimiser import HOME_CELL, START, _grid

TOLERANCE = 0.05
MIN_DEPTH = 25.0


@pytest.fixture(scope="module")
def graph():
    return _grid()


def _hex_of(graph, node):
    x, y = _TO_BNG.transform(graph.nodes[node]["x"], graph.nodes[node]["y"])
    return bng_to_hex(x, y), x, y


# --- step 1: anchors ---------------------------------------------------------------


def test_anchors_are_safe_nodes_inside_their_hexagon(graph):
    anchors = find_anchors(graph, MIN_DEPTH)
    assert anchors
    for hex_id, by_direction in anchors.items():
        assert set(by_direction) == set(neighbours(hex_id))
        for node in by_direction.values():
            found, x, y = _hex_of(graph, node)
            assert found == hex_id
            assert depth_in_hex(x, y, hex_id) >= MIN_DEPTH


def test_anchor_is_the_safe_node_closest_to_the_edge_midpoint(graph):
    anchors = find_anchors(graph, MIN_DEPTH)
    hex_id = _hex_of(graph, START)[0]
    cx, cy = hex_centre_bng(hex_id)
    for direction, anchor in anchors[hex_id].items():
        dx, dy = hex_centre_bng(direction)
        mid = ((cx + dx) / 2, (cy + dy) / 2)
        safe = [
            n
            for n in graph.nodes
            if _hex_of(graph, n)[0] == hex_id
            and depth_in_hex(*_hex_of(graph, n)[1:], hex_id) >= MIN_DEPTH
        ]

        def gap(n):
            _, x, y = _hex_of(graph, n)
            return (x - mid[0]) ** 2 + (y - mid[1]) ** 2

        assert gap(anchor) == pytest.approx(min(gap(n) for n in safe))


def test_hexagons_without_a_safe_node_have_no_anchors():
    g = nx.MultiDiGraph()
    cx, cy = hex_centre_bng(HOME_CELL)
    ox, oy = hex_centre_bng(neighbours(HOME_CELL)[0])
    mx, my = (cx + ox) / 2, (cy + oy) / 2  # the middle of the edge they share
    reach = ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5
    scale = 1 - 5 / reach  # 5 m short of that edge
    lon, lat = _TO_WGS84.transform(cx + (mx - cx) * scale, cy + (my - cy) * scale)
    g.add_node(1, x=lon, y=lat)
    assert find_anchors(g, MIN_DEPTH) == {}
    assert HOME_CELL in find_anchors(g, 0)


def test_anchors_can_be_limited_to_some_hexagons(graph):
    hex_id = _hex_of(graph, START)[0]
    assert set(find_anchors(graph, MIN_DEPTH, {hex_id})) == {hex_id}


# --- steps 2-4: the planned route ----------------------------------------------------


@pytest.mark.parametrize("target_m", [3000, 4000, 6000])
def test_planned_route_is_closed_within_tolerance_and_counts_safe_hexagons(graph, target_m):
    planned = plan_route(graph, START, target_m)

    route = planned.route
    assert route[0] == route[-1] == START
    length = route_length(graph, route)
    assert abs(length - target_m) <= TOLERANCE * target_m
    assert planned.length_m == pytest.approx(length)

    # Only safe hexagons count: the figure agrees with the scorer and with the map.
    score = score_route(graph, route, min_depth_m=MIN_DEPTH)
    assert planned.unique_hexes == score.unique_hexes
    assert planned.unique_hexes == len(hexes_for_track(route_to_latlons(graph, route), MIN_DEPTH))
    assert planned.wasted_m == pytest.approx(score.wasted_m)
    assert planned.unique_hexes > 1


@pytest.mark.parametrize("target_m", [3000, 4000, 6000])
def test_planner_covers_at_least_as_many_safe_hexagons_as_the_random_search(graph, target_m):
    planned = plan_route(graph, START, target_m)
    random = optimise_route(graph, START, target_m, n_candidates=50, seed=0, verbose=False)
    assert planned.unique_hexes >= random.unique_hexes


def test_planning_is_deterministic(graph):
    assert plan_route(graph, START, 4000).route == plan_route(graph, START, 4000).route


def test_radius_limits_the_hexagons_planned(graph):
    wide = plan_route(graph, START, 6000)
    near = plan_route(graph, START, 6000, radius_m=1000)
    centre = _hex_of(graph, START)[0]
    for node in near.route:
        assert hex_distance(_hex_of(graph, node)[0], centre) <= 3
    assert wide.unique_hexes >= 1


def test_planner_fails_when_no_plan_fits_the_distance(graph):
    with pytest.raises(PlannerError):
        plan_route(graph, START, 300_000)
    with pytest.raises(PlannerError):
        plan_route(graph, START, 100)  # too short to leave the start hexagon


def test_planner_times_out(graph):
    with pytest.raises(PlannerTimeout):
        plan_route(graph, START, 4000, time_limit_s=-1)


def test_invalid_arguments(graph):
    with pytest.raises(ValueError):
        plan_route(graph, (999, 999), 4000)
    with pytest.raises(ValueError):
        plan_route(graph, START, 0)
    with pytest.raises(ValueError):
        plan_route(graph, START, 4000, beam_width=0)
    with pytest.raises(ValueError):
        plan_route(graph, START, 4000, min_depth_m=-1)


def test_border_hops_are_preferred():
    """A step into a cell that touches the previous one ranks as cheaper than the same distance elsewhere."""
    first, second = neighbours(HOME_CELL)[:2]
    far = next(c for c in hexes_within_radius(51.5, -0.12, 3000) if hex_distance(c, HOME_CELL) == 2)
    assert planner._adjacent(HOME_CELL, first)
    assert planner._adjacent(first, HOME_CELL)
    assert not planner._adjacent(HOME_CELL, far)
    assert planner.BORDER_HOP_WEIGHT < 1


# --- cells you already own -----------------------------------------------------------


def _cells_of(graph, route):
    return hexes_for_track(route_to_latlons(graph, route), MIN_DEPTH)


def test_claimed_cells_are_not_counted_as_new(graph):
    plain = plan_route(graph, START, 4000)
    claimed = frozenset(list(_cells_of(graph, plain.route))[:2])
    planned = plan_route(graph, START, 4000, claimed=claimed)

    # What the planner reports is what the map and the scorer say, with the claimed cells taken out.
    score = score_route(graph, planned.route, min_depth_m=MIN_DEPTH, claimed=claimed)
    assert planned.unique_hexes == score.unique_hexes
    assert planned.unique_hexes == len(_cells_of(graph, planned.route) - claimed)


def test_the_planner_prefers_unclaimed_cells(graph):
    plain = plan_route(graph, START, 4000)
    claimed = frozenset(_cells_of(graph, plain.route)) - {_hex_of(graph, START)[0]}
    planned = plan_route(graph, START, 4000, claimed=claimed)

    assert planned.route[0] == planned.route[-1] == START
    assert abs(route_length(graph, planned.route) - 4000) <= TOLERANCE * 4000
    # Owning everything the first plan covered must push the new plan onto fresh cells.
    assert planned.unique_hexes >= 1
    assert len(_cells_of(graph, planned.route) - claimed) == planned.unique_hexes


# --- planner versus random search ----------------------------------------------------


def test_choose_route_returns_the_better_of_the_two(graph):
    choice = choose_route(graph, START, 6000, n_candidates=30)
    other = choice.random if choice.chosen is choice.planner else choice.planner
    assert choice.chosen.safe_hexes >= other.safe_hexes
    assert choice.chosen.route[0] == choice.chosen.route[-1] == START
    assert abs(choice.chosen.length_m - 6000) <= TOLERANCE * 6000
    assert choice.planner.seconds >= 0 and choice.random.seconds >= 0


def test_choose_route_keeps_the_random_result_when_it_has_more_safe_hexagons(graph, monkeypatch):
    # The random search is scored first, then the planner: give them 5 and 2 safe hexagons.
    counts = iter([5, 2])
    monkeypatch.setattr(planner, "hexes_for_track", lambda points, depth: set(range(next(counts))))
    choice = choose_route(graph, START, 4000, n_candidates=20)
    assert (choice.random.safe_hexes, choice.planner.safe_hexes) == (5, 2)
    assert choice.chosen is choice.random
    assert choice.reason == "more safe hexagons"


def test_choose_route_keeps_the_planner_result_when_it_has_more_safe_hexagons(graph, monkeypatch):
    counts = iter([2, 5])
    monkeypatch.setattr(planner, "hexes_for_track", lambda points, depth: set(range(next(counts))))
    choice = choose_route(graph, START, 4000, n_candidates=20)
    assert choice.chosen is choice.planner


def test_fallback_when_the_planner_fails(graph, monkeypatch):
    def boom(*args, **kwargs):
        raise PlannerError("no plan")

    monkeypatch.setattr(planner, "plan_route", boom)
    choice = choose_route(graph, START, 4000, n_candidates=20)
    assert choice.chosen is choice.random
    assert choice.planner.route is None and choice.planner.error == "no plan"
    assert "planner failed" in choice.reason
    assert choice.chosen.route[0] == choice.chosen.route[-1] == START


def test_fallback_when_the_planner_crashes(graph, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(planner, "plan_route", boom)
    choice = choose_route(graph, START, 4000, n_candidates=20)
    assert choice.chosen is choice.random
    assert "RuntimeError" in choice.planner.error


def test_fallback_when_the_planner_is_too_slow(graph, monkeypatch):
    real = planner.plan_route

    def slow(*args, **kwargs):
        kwargs["time_limit_s"] = None  # finishes, but takes longer than the limit
        return real(*args, **kwargs)

    monkeypatch.setattr(planner, "plan_route", slow)
    choice = choose_route(graph, START, 4000, n_candidates=20, time_limit_s=0.0)
    assert choice.chosen is choice.random
    assert "limit" in choice.planner.error


def test_planner_result_is_used_when_the_random_search_fails(graph, monkeypatch):
    def boom(*args, **kwargs):
        raise LoopNotFoundError("none")

    monkeypatch.setattr(planner, "optimise_route", boom)
    choice = choose_route(graph, START, 4000)
    assert choice.chosen is choice.planner
    assert choice.random.error == "none"


def test_both_failing_raises(graph, monkeypatch):
    def boom(*args, **kwargs):
        raise LoopNotFoundError("none")

    monkeypatch.setattr(planner, "optimise_route", boom)
    monkeypatch.setattr(planner, "plan_route", boom)
    with pytest.raises(LoopNotFoundError):
        choose_route(graph, START, 4000)


def test_format_choice_shows_both_side_by_side(graph, monkeypatch):
    text = format_choice(choose_route(graph, START, 4000, n_candidates=20))
    for word in ("planner", "random search", "safe hexagons", "length", "wasted distance", "time taken", "chosen:"):
        assert word in text

    monkeypatch.setattr(planner, "plan_route", lambda *a, **k: (_ for _ in ()).throw(PlannerError("x")))
    failed = format_choice(choose_route(graph, START, 4000, n_candidates=20))
    assert "failed" in failed and "chosen: random search" in failed


# --- planner only (the web app) --------------------------------------------------------


def _fail_if_called(name):
    def boom(*args, **kwargs):
        raise AssertionError(f"{name} must not run")

    return boom


def test_fast_route_never_runs_the_random_search_when_the_planner_works(graph, monkeypatch):
    monkeypatch.setattr(planner, "optimise_route", _fail_if_called("optimise_route"))
    choice = choose_route_fast(graph, START, 4000)

    assert choice.chosen is choice.planner
    assert choice.chosen.route[0] == choice.chosen.route[-1] == START
    assert abs(choice.chosen.length_m - 4000) <= TOLERANCE * 4000
    assert choice.random.route is None and choice.random.error == SKIPPED
    assert choice.reason == "planner only"


def test_fast_route_reports_its_stages_in_order(graph):
    stages = []
    choose_route_fast(graph, START, 4000, on_stage=stages.append)
    assert stages == [STAGE_CHOOSING, STAGE_JOINING]


def test_fast_route_falls_back_to_the_random_search_up_to_12_km(graph, monkeypatch):
    monkeypatch.setattr(planner, "plan_route", lambda *a, **k: (_ for _ in ()).throw(PlannerError("no plan")))
    stages = []
    choice = choose_route_fast(graph, START, 4000, n_candidates=20, on_stage=stages.append)

    assert choice.chosen is choice.random
    assert choice.planner.error == "no plan"
    assert "planner failed" in choice.reason
    assert choice.chosen.route[0] == choice.chosen.route[-1] == START
    assert stages == [STAGE_FALLBACK]


def test_fast_route_falls_back_when_the_planner_crashes_or_is_too_slow(graph, monkeypatch):
    monkeypatch.setattr(planner, "plan_route", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("oops")))
    assert choose_route_fast(graph, START, 4000, n_candidates=20).chosen.name == "random search"

    monkeypatch.undo()
    real = planner.plan_route
    monkeypatch.setattr(planner, "plan_route", lambda *a, **k: real(*a, **{**k, "time_limit_s": None}))
    slow = choose_route_fast(graph, START, 4000, n_candidates=20, time_limit_s=0.0)
    assert slow.chosen.name == "random search" and "limit" in slow.planner.error


def test_fast_route_falls_back_at_exactly_12_km_but_not_above(graph, monkeypatch):
    monkeypatch.setattr(planner, "plan_route", lambda *a, **k: (_ for _ in ()).throw(PlannerError("no plan")))
    ran = []
    monkeypatch.setattr(planner, "_random_search", lambda *a, **k: ran.append(a[2]) or SearchResult(
        "random search", 0.0, route=[START, START], polished=object()))

    assert choose_route_fast(graph, START, 12_000).chosen.name == "random search"
    assert ran == [12_000]

    with pytest.raises(LoopNotFoundError, match="only a fallback up to 12 km"):
        choose_route_fast(graph, START, 12_001)
    assert ran == [12_000]  # the random search was not tried for the longer route


def test_fast_route_fails_when_the_fallback_fails_too(graph, monkeypatch):
    monkeypatch.setattr(planner, "plan_route", lambda *a, **k: (_ for _ in ()).throw(PlannerError("no plan")))
    monkeypatch.setattr(planner, "optimise_route", lambda *a, **k: (_ for _ in ()).throw(LoopNotFoundError("none")))
    with pytest.raises(LoopNotFoundError, match="no plan"):
        choose_route_fast(graph, START, 4000)


def test_format_choice_says_skipped_for_the_search_that_did_not_run(graph):
    text = format_choice(choose_route_fast(graph, START, 4000))
    assert "skipped" in text and "chosen: planner (planner only)" in text
    assert "failed" not in text
