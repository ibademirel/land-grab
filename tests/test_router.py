import networkx as nx
import pytest

from land_grab.router import LoopNotFoundError, make_loop, route_length

BLOCK_M = 100
START = (10, 10)
TARGET_M = 4000


def _grid(n=21, block_m=BLOCK_M):
    """An n x n street grid, two-way like OSMnx walk networks, with 100 m blocks."""
    g = nx.MultiDiGraph()
    for u, v in nx.grid_2d_graph(n, n).edges():
        g.add_edge(u, v, length=block_m)
        g.add_edge(v, u, length=block_m)
    return g


def _streets(route):
    return [frozenset(edge) for edge in zip(route, route[1:])]


def _return_reuse(graph, route):
    """Fraction of the return half's streets that were already used on the way out.

    The route is split where its real length passes half the total.
    """
    total = route_length(graph, route)
    walked, split = 0.0, 0
    while walked < total / 2:
        walked += route_length(graph, route[split : split + 2])
        split += 1
    outward = set(_streets(route[: split + 1]))
    back = _streets(route[split:])
    return sum(street in outward for street in back) / len(back)


def test_loop_returns_to_start_and_is_within_tolerance():
    g = _grid()
    route = make_loop(g, START, TARGET_M, seed=1)

    assert route[0] == START
    assert route[-1] == START
    assert abs(route_length(g, route) - TARGET_M) <= 0.05 * TARGET_M


@pytest.mark.parametrize("seed", range(20))
def test_loops_for_many_seeds_close_and_hit_target(seed):
    g = _grid()
    route = make_loop(g, START, TARGET_M, seed=seed)
    assert route[0] == route[-1] == START
    assert abs(route_length(g, route) - TARGET_M) <= 0.05 * TARGET_M


def test_route_is_a_connected_walk_on_the_graph():
    g = _grid()
    route = make_loop(g, START, TARGET_M, seed=3)
    assert all(g.has_edge(u, v) for u, v in zip(route, route[1:]))


@pytest.mark.parametrize("seed", range(20))
def test_return_leg_does_not_retrace_the_way_out(seed):
    g = _grid()
    route = make_loop(g, START, TARGET_M, seed=seed)
    assert _return_reuse(g, route) <= 0.1


def test_penalty_is_what_prevents_retracing():
    # Without the penalty (penalty=1) some seeds come back the way they went out.
    g = _grid()
    with_penalty = [_return_reuse(g, make_loop(g, START, TARGET_M, seed=s)) for s in range(30)]
    without = [
        _return_reuse(g, make_loop(g, START, TARGET_M, seed=s, penalty=1.0)) for s in range(30)
    ]
    assert sum(with_penalty) < sum(without)
    assert max(without) > 0.2


def test_reported_length_is_real_not_penalised():
    g = _grid()
    route = make_loop(g, START, TARGET_M, seed=5)
    # Every block is 100 m, so the real length is simply the number of steps.
    assert route_length(g, route) == BLOCK_M * (len(route) - 1)


def test_same_seed_gives_same_route():
    g = _grid()
    assert make_loop(g, START, TARGET_M, seed=42) == make_loop(g, START, TARGET_M, seed=42)


def test_works_on_a_simple_undirected_graph():
    g = nx.Graph()
    for u, v in nx.grid_2d_graph(21, 21).edges():
        g.add_edge(u, v, length=BLOCK_M)
    route = make_loop(g, START, TARGET_M, seed=7)
    assert route[0] == route[-1] == START
    assert abs(route_length(g, route) - TARGET_M) <= 0.05 * TARGET_M


def test_raises_when_target_is_impossible():
    # A 5 x 5 grid has 4 km of streets in total; a 50 km loop can never fit.
    with pytest.raises(LoopNotFoundError, match="after 200 tries"):
        make_loop(_grid(n=5), (2, 2), 50_000, seed=1)


def test_raises_when_start_cannot_reach_anything():
    g = _grid(n=5)
    g.add_node("island")
    with pytest.raises(LoopNotFoundError, match="no loop could be routed"):
        make_loop(g, "island", 1000, seed=1, max_tries=5)


def test_invalid_arguments_raise_value_error():
    g = _grid(n=5)
    with pytest.raises(ValueError, match="not in the graph"):
        make_loop(g, (99, 99), 1000, seed=1)
    with pytest.raises(ValueError):
        make_loop(g, (2, 2), 0, seed=1)
    with pytest.raises(ValueError):
        make_loop(g, (2, 2), 1000, seed=1, max_tries=0)
