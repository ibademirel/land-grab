"""Pick the loop that claims the most new H3 cells for its distance."""

import math
import random
from dataclasses import dataclass

import networkx as nx

from land_grab.coverage import (
    DEFAULT_MIN_DEPTH_M,
    _densify_bng,
    _meets_margin,
    _validate_min_depth,
)
from land_grab.hexgrid import _TO_BNG, HexId, bng_to_hex, depth_in_hex
from land_grab.router import DEFAULT_TOLERANCE, LoopNotFoundError, make_loop, route_length


@dataclass(frozen=True)
class RouteScore:
    unique_hexes: int  # new cells claimed: each counts once (cells you already own do not count)
    wasted_m: float  # metres spent re-entering hexagons already claimed earlier


@dataclass(frozen=True)
class OptimisedRoute:
    route: list  # nodes, start to start
    unique_hexes: int
    length_m: float
    wasted_m: float
    candidates_tried: int
    candidates_kept: int


def _edge_data(graph: nx.Graph, u, v) -> dict:
    """Data of the edge u -> v; for parallel edges the shortest, as route_length uses."""
    data = graph[u][v]
    if graph.is_multigraph():
        return min(data.values(), key=lambda attrs: attrs["length"])
    return data


def _edge_pieces(graph: nx.Graph, u, v) -> list[tuple[HexId, float, float]]:
    """The edge u -> v as (hex_id, metres, depth_m) pieces of at most 20 m, in travel order.

    Uses the edge's geometry if it has one (oriented to run from u to v),
    otherwise the straight line between the nodes. Each piece is assigned to
    the H3 cell containing its midpoint, and ``depth_m`` is how far inside that
    cell the midpoint is (distance to the cell's boundary).
    """
    geometry = _edge_data(graph, u, v).get("geometry")
    if geometry is not None:
        points = [(lat, lon) for lon, lat in geometry.coords]
        start = (graph.nodes[u]["y"], graph.nodes[u]["x"])
        if math.dist(points[-1], start) < math.dist(points[0], start):
            points.reverse()
    else:
        points = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in (u, v)]

    dense = _densify_bng(points)
    pieces = []
    for (x0, y0), (x1, y1) in zip(dense, dense[1:]):
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        hex_id = bng_to_hex(mid_x, mid_y)
        pieces.append((hex_id, math.hypot(x1 - x0, y1 - y0), depth_in_hex(mid_x, mid_y, hex_id)))
    return pieces


def score_route(
    graph: nx.Graph,
    route: list,
    _piece_cache: dict | None = None,
    min_depth_m: float = DEFAULT_MIN_DEPTH_M,
    claimed: frozenset = frozenset(),
) -> RouteScore:
    """Score a route by the Land Grab rule: each hexagon counts once, repeats count zero.

    ``unique_hexes`` is the number of distinct hexagons the route passes through.
    ``wasted_m`` is the distance spent in hexagons that were already claimed on
    an earlier visit. A visit is a continuous stretch inside one hexagon, so
    walking on through the hexagon you are in is not wasted; leaving it and
    coming back later is. An out-and-back route wastes almost its whole return
    leg; a loop whose return runs through other hexagons wastes nothing.

    GPS safety margin: a hexagon only counts towards ``unique_hexes`` if the
    route gets at least ``min_depth_m`` inside it, so routes that only clip an
    edge or corner score lower than routes that go well inside. (``wasted_m``
    still treats every hexagon the route enters as claimed.) ``min_depth_m=0``
    counts every hexagon entered.

    ``claimed`` is a set of cell IDs you already own. They are not new points:
    they never count towards ``unique_hexes``, however deep the route goes.
    """
    _validate_min_depth(min_depth_m)
    cache = {} if _piece_cache is None else _piece_cache

    visited: set[HexId] = set()  # cells entered so far (for the wasted distance)
    deep_enough: set[HexId] = set()
    wasted = 0.0
    current: HexId | None = None
    current_is_repeat = False
    for u, v in zip(route, route[1:]):
        if (u, v) not in cache:
            cache[(u, v)] = _edge_pieces(graph, u, v)
        for hex_id, metres, depth_m in cache[(u, v)]:
            if hex_id not in claimed and _meets_margin(depth_m, min_depth_m):
                deep_enough.add(hex_id)
            if hex_id != current:
                current = hex_id
                current_is_repeat = hex_id in visited
                visited.add(hex_id)
            if current_is_repeat:
                wasted += metres

    if len(route) == 1:
        node = graph.nodes[route[0]]
        x, y = _TO_BNG.transform(node["x"], node["y"])
        hex_id = bng_to_hex(x, y)
        if hex_id not in claimed and _meets_margin(depth_in_hex(x, y, hex_id), min_depth_m):
            deep_enough.add(hex_id)
    return RouteScore(unique_hexes=len(deep_enough), wasted_m=wasted)


def optimise_route(
    graph: nx.Graph,
    start_node,
    target_m: float,
    n_candidates: int = 300,
    seed: int = 0,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    min_depth_m: float = DEFAULT_MIN_DEPTH_M,
    claimed: frozenset = frozenset(),
    verbose: bool = True,
) -> OptimisedRoute:
    """Generate candidate loops with make_loop and return the one claiming the most hexagons.

    Each of the ``n_candidates`` loops is built with a different seed (derived
    deterministically from ``seed``). Candidates whose real length is not within
    ``tolerance`` of ``target_m`` are discarded, as are seeds for which
    make_loop finds no loop. The kept candidates are ranked by unique hexagons
    (higher is better), then by wasted distance (lower is better), then by
    closeness to the target length. Identical loops from different seeds are
    scored once. Cells in ``claimed`` are not counted as new. Hexagons count only if the route gets ``min_depth_m`` inside
    them (see ``score_route``), so routes that go well inside hexagons win.

    Prints a short summary unless ``verbose`` is False. Raises
    LoopNotFoundError if no candidate is kept.
    """
    _validate_min_depth(min_depth_m)
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be at least 1, got {n_candidates}")

    rng = random.Random(seed)
    piece_cache: dict = {}
    scored: dict[tuple, tuple[RouteScore, float]] = {}
    kept = 0
    for _ in range(n_candidates):
        try:
            route = make_loop(
                graph, start_node, target_m, seed=rng.getrandbits(32), tolerance=tolerance
            )
        except LoopNotFoundError:
            continue
        length = route_length(graph, route)
        if abs(length - target_m) > tolerance * target_m:
            continue
        kept += 1
        key = tuple(route)
        if key not in scored:
            scored[key] = (score_route(graph, route, piece_cache, min_depth_m, claimed), length)

    if not scored:
        raise LoopNotFoundError(
            f"None of {n_candidates} candidates was within {tolerance:.0%} of {target_m:.0f} m"
        )

    best_key, (best_score, best_length) = min(
        scored.items(),
        key=lambda item: (
            -item[1][0].unique_hexes,
            item[1][0].wasted_m,
            abs(item[1][1] - target_m),
        ),
    )
    result = OptimisedRoute(
        route=list(best_key),
        unique_hexes=best_score.unique_hexes,
        length_m=best_length,
        wasted_m=best_score.wasted_m,
        candidates_tried=n_candidates,
        candidates_kept=kept,
    )
    if verbose:
        print(
            f"candidates tried: {n_candidates} (kept {kept}, {len(scored)} distinct)\n"
            f"best score: {result.unique_hexes} hexagons\n"
            f"best length: {result.length_m / 1000:.2f} km\n"
            f"wasted distance: {result.wasted_m / 1000:.2f} km"
        )
    return result
