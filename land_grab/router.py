"""Generate closed loop routes on a street network."""

import math
import random

import networkx as nx

DEFAULT_TOLERANCE = 0.05
DEFAULT_MAX_TRIES = 200
DEFAULT_PENALTY = 5.0

# Waypoints are picked from nodes whose network distance from the start is
# within this band around the target ring radius.
_RING_LOW, _RING_HIGH = 0.75, 1.25


class LoopNotFoundError(RuntimeError):
    """No loop within tolerance of the target length was found."""


def _edge_length(graph: nx.Graph, u, v) -> float:
    """Real length of the shortest edge from u to v (graphs may have parallel edges)."""
    data = graph[u][v]
    if graph.is_multigraph():
        return min(attrs["length"] for attrs in data.values())
    return data["length"]


def route_length(graph: nx.Graph, route: list) -> float:
    """Return the real length in metres of a route given as a list of nodes."""
    return sum(_edge_length(graph, u, v) for u, v in zip(route, route[1:]))


def _penalised_weight(graph: nx.Graph, used: set[frozenset], penalty: float):
    """Routing weight: edge length, multiplied by `penalty` if an earlier leg used that street.

    Streets are compared without direction, so walking an earlier leg's edge
    backwards is penalised too. That is what stops out-and-back routes.
    """
    multigraph = graph.is_multigraph()

    def weight(u, v, data):
        length = (
            min(attrs["length"] for attrs in data.values()) if multigraph else data["length"]
        )
        return length * penalty if frozenset((u, v)) in used else length

    return weight


def _ring(distances: dict, radius: float, exclude: set) -> list:
    return [
        node
        for node, dist in distances.items()
        if _RING_LOW * radius <= dist <= _RING_HIGH * radius and node not in exclude
    ]


def _try_loop(
    graph: nx.Graph, start_node, radius: float, rng: random.Random, penalty: float
) -> list | None:
    """Build one candidate loop start -> w1 -> w2 -> start, or None if it can't be routed."""
    cutoff = _RING_HIGH * radius
    from_start = nx.single_source_dijkstra_path_length(
        graph, start_node, cutoff=cutoff, weight="length"
    )
    ring = _ring(from_start, radius, {start_node})
    if not ring:
        return None
    w1 = rng.choice(ring)

    # Prefer a second waypoint that is also about `radius` from w1, giving a
    # roughly equilateral triangle with a perimeter close to 3 * radius.
    from_w1 = nx.single_source_dijkstra_path_length(graph, w1, cutoff=cutoff, weight="length")
    ring_set = set(ring)
    candidates = [node for node in _ring(from_w1, radius, {start_node, w1}) if node in ring_set]
    if not candidates:
        candidates = [node for node in ring if node != w1]
    if not candidates:
        return None
    w2 = rng.choice(candidates)

    used: set[frozenset] = set()
    route = [start_node]
    for source, target in ((start_node, w1), (w1, w2), (w2, start_node)):
        try:
            leg = nx.shortest_path(
                graph, source, target, weight=_penalised_weight(graph, used, penalty)
            )
        except nx.NetworkXNoPath:
            return None
        used.update(frozenset(edge) for edge in zip(leg, leg[1:]))
        route.extend(leg[1:])
    return route


def make_loop(
    graph: nx.Graph,
    start_node,
    target_m: float,
    seed: int | None = None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    max_tries: int = DEFAULT_MAX_TRIES,
    penalty: float = DEFAULT_PENALTY,
) -> list:
    """Return a closed route (list of nodes, start to start) within `tolerance` of `target_m`.

    Each try picks two random waypoints about ``target_m / 3`` (network
    distance) from the start, then routes start -> w1 -> w2 -> start with
    shortest paths. While routing each leg, streets used by earlier legs cost
    ``penalty`` times their length, so the way back takes a different street
    instead of retracing the way out. The penalty only affects routing: the
    length checked against the target is the real length (see
    ``route_length``).

    Failed tries are retried with new random waypoints. The ring radius is also
    rescaled by target / actual length after each try, since penalised detours
    make loops longer than the waypoint triangle. Results are deterministic for
    a given ``seed``. Edges need a ``"length"`` attribute in metres (as OSMnx
    provides).

    Raises ValueError for bad arguments, and LoopNotFoundError if no loop
    within tolerance is found after ``max_tries`` tries.
    """
    if start_node not in graph:
        raise ValueError(f"start_node {start_node!r} is not in the graph")
    if target_m <= 0:
        raise ValueError(f"target_m must be positive, got {target_m}")
    if max_tries < 1:
        raise ValueError(f"max_tries must be at least 1, got {max_tries}")

    rng = random.Random(seed)
    scale = 1.0
    closest = math.inf
    for _ in range(max_tries):
        route = _try_loop(graph, start_node, scale * target_m / 3, rng, penalty)
        if route is None:
            continue
        length = route_length(graph, route)
        if abs(length - target_m) <= tolerance * target_m:
            return route
        closest = min(closest, length, key=lambda x: abs(x - target_m))
        scale = min(3.0, max(0.2, scale * min(1.5, max(0.67, target_m / length))))

    found = f"closest was {closest:.0f} m" if closest < math.inf else "no loop could be routed"
    raise LoopNotFoundError(
        f"No loop within {tolerance:.0%} of {target_m:.0f} m after {max_tries} tries ({found})"
    )
