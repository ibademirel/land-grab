"""Hexagon-first route planner: choose which H3 cells to visit, then link them with streets.

Instead of making random loops and keeping the best (``optimise_route``), the
planner works at cell level:

1. Safe anchor points: in every H3 cell, the graph nodes at least
   ``min_depth_m`` inside it (measured to the cell's real boundary) are safe.
   For each of the cell's edges the safe node closest to that edge's midpoint
   is an anchor, so stepping into the neighbouring cell across it is cheap.
2. A beam search builds closed walks through neighbouring hexagons, starting
   from the start hexagon and never visiting a hexagon twice. Steps cost the
   real street distance between anchors.
3. Plans are closed (the shortest way home) and kept if their length is within
   ``target_m`` +/- ``tolerance``; more new hexagons win, then closeness to the target.
   Cells you already own (``claimed``) are not new: a plan may cross one as a
   stepping stone, but each ranks as longer and adds nothing to the count.
4. The chosen anchors are joined by shortest paths with the street-reuse
   penalty, and the street route is polished with ``polish_route``.

``choose_route`` runs both this planner and the random search and keeps the better one.
``choose_route_fast`` runs only the planner, with the random search as a fallback for short routes.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import networkx as nx
import numpy as np

from land_grab.coverage import (
    DEFAULT_MIN_DEPTH_M,
    _meets_margin,
    _validate_min_depth,
    hexes_for_track,
)
from land_grab.hexgrid import (
    _TO_BNG,
    HexId,
    bng_to_hex,
    depth_in_hex,
    hex_centre_bng,
    hexes_within_radius,
    neighbours,
)
from land_grab.mapview import route_to_latlons
from land_grab.optimiser import optimise_route, score_route
from land_grab.polish import PolishedRoute, polish_route
from land_grab.router import (
    DEFAULT_PENALTY,
    DEFAULT_TOLERANCE,
    LoopNotFoundError,
    _penalised_weight,
    route_length,
)

DEFAULT_BEAM_WIDTH = 30
DEFAULT_CUTOFF_M = 2000.0  # longest street distance looked at for one step between anchors
DEFAULT_TIME_LIMIT_S = 90.0
BORDER_HOP_WEIGHT = 0.8  # a step along the border between two rows of hexagons ranks as 20% shorter
CLAIMED_STEP_WEIGHT = 3.0  # a step into a cell you already own ranks as 3 times longer: unclaimed cells are preferred
RETURN_WEIGHT = 0.5  # how much the distance home counts when ranking partial plans
MAX_BUILDS = 5  # best finished plans per first direction that are turned into street routes
FALLBACK_MAX_M = 12_000.0  # choose_route_fast: the random search is only a fallback up to this length

# Progress stages reported through ``on_stage`` (shown to people, so plain words).
STAGE_CHOOSING = "Choosing hexagons"
STAGE_JOINING = "Joining them up"
STAGE_FALLBACK = "Trying another way"
SKIPPED = "skipped"  # SearchResult.error of a search that was not run on purpose

_EPS = 1e-6

Anchors = dict[HexId, dict[HexId, object]]  # cell -> {neighbouring cell across the edge: node}


class PlannerError(LoopNotFoundError):
    """The planner found no route within tolerance."""


class PlannerTimeout(PlannerError):
    """The planner ran out of time."""


def _adjacent(a: HexId, b: HexId) -> bool:
    return a in neighbours(b)


def find_anchors(
    graph: nx.Graph,
    min_depth_m: float = DEFAULT_MIN_DEPTH_M,
    hex_ids: set[HexId] | None = None,
) -> Anchors:
    """Return the anchor nodes of every hexagon that has a safe node.

    A node is safe in its cell if it is at least ``min_depth_m`` inside
    (distance to the nearest edge of the H3 cell). For each edge, keyed by
    the ID of the neighbouring cell across it, the anchor is the safe node
    closest to the edge's midpoint. One node can be the anchor of several
    edges. Cells without a safe node have no anchors. ``hex_ids`` limits the
    result to those cells.
    """
    _validate_min_depth(min_depth_m)
    nodes = list(graph.nodes)
    if not nodes:
        return {}
    eastings, northings = _TO_BNG.transform(
        [graph.nodes[n]["x"] for n in nodes], [graph.nodes[n]["y"] for n in nodes]
    )

    safe: dict[HexId, list] = {}
    for node, x, y in zip(nodes, eastings, northings):
        hex_id = bng_to_hex(x, y)
        if hex_ids is not None and hex_id not in hex_ids:
            continue
        if _meets_margin(depth_in_hex(x, y, hex_id), min_depth_m):
            safe.setdefault(hex_id, []).append((node, x, y))

    anchors: Anchors = {}
    for hex_id, found in safe.items():
        cx, cy = hex_centre_bng(hex_id)
        points = np.array([(x, y) for _, x, y in found])
        anchors[hex_id] = {}
        for other in neighbours(hex_id):
            # The shared edge's midpoint is halfway to the neighbour's centre.
            ox, oy = hex_centre_bng(other)
            gaps = np.hypot(points[:, 0] - (cx + ox) / 2, points[:, 1] - (cy + oy) / 2)
            anchors[hex_id][other] = found[int(np.argmin(gaps))][0]
    return anchors


@dataclass(frozen=True)
class _Plan:
    hexes: tuple  # hexagons in visiting order, start hexagon first
    nodes: tuple  # the start node, then the anchor used in each later hexagon
    length_m: float  # street distance so far (shortest paths between the nodes)
    rank_m: float  # the same, with border hops weighted down
    visited: frozenset


@dataclass(frozen=True)
class PlannedRoute:
    route: list  # nodes, start to start, polished
    unique_hexes: int  # new safe hexagons (score_route, with min_depth_m; claimed cells not counted)
    length_m: float
    wasted_m: float
    first_hex: HexId  # the start hexagon's neighbour the chosen plan went to first
    planned_hexes: int  # hexagons in the hexagon-level plan (anchors visited, start included)
    directions_tried: int  # first directions that gave a street route within tolerance
    polished: PolishedRoute


class _Planner:
    def __init__(
        self, graph, start_node, target_m, tolerance, min_depth_m, hex_ids,
        beam_width, cutoff_m, penalty, deadline, claimed,
    ):
        self.graph = graph
        self.start_node = start_node
        self.target_m = target_m
        self.claimed = claimed
        self.min_depth_m = min_depth_m
        self.low_m = target_m * (1 - tolerance)
        self.high_m = target_m * (1 + tolerance)
        self.beam_width = beam_width
        self.cutoff_m = cutoff_m
        self.penalty = penalty
        self.deadline = deadline
        self.pieces: dict = {}
        self._from: dict = {}

        node = graph.nodes[start_node]
        x, y = _TO_BNG.transform(node["x"], node["y"])
        self.start_hex = bng_to_hex(x, y)
        self.anchors = find_anchors(graph, min_depth_m, hex_ids)
        reverse = graph.reverse(copy=False) if graph.is_directed() else graph
        # Shortest street distance from each node back to the start.
        self.home = nx.single_source_dijkstra_path_length(
            reverse, start_node, cutoff=self.high_m, weight="length"
        )

    def check_time(self) -> None:
        if self.deadline is not None and time.perf_counter() > self.deadline:
            raise PlannerTimeout("the planner ran out of time")

    def distances_from(self, node) -> dict:
        if node not in self._from:
            self._from[node] = nx.single_source_dijkstra_path_length(
                self.graph, node, cutoff=self.cutoff_m, weight="length"
            )
        return self._from[node]

    def _rank(self, plan: _Plan) -> float:
        return plan.rank_m + RETURN_WEIGHT * self.home[plan.nodes[-1]]

    def _steps(self, plan: _Plan, next_hexes):
        """Plans one hexagon longer, for each reachable anchor of each unvisited hexagon."""
        dist = self.distances_from(plan.nodes[-1])
        previous = plan.hexes[-2] if len(plan.hexes) >= 2 else None
        for hex_id in next_hexes:
            if hex_id in plan.visited or hex_id not in self.anchors:
                continue
            # Next to both this hexagon and the one before: hopping along the
            # border between two rows, which gains hexagons for the least distance.
            weight = BORDER_HOP_WEIGHT if previous is not None and _adjacent(previous, hex_id) else 1.0
            if hex_id in self.claimed:
                weight *= CLAIMED_STEP_WEIGHT  # allowed as a stepping stone, but unclaimed cells are preferred
            for anchor in set(self.anchors[hex_id].values()):
                step = dist.get(anchor)
                home = self.home.get(anchor)
                # Only if the shortest way home still fits in the remaining distance.
                if step is None or home is None or plan.length_m + step + home > self.high_m + _EPS:
                    continue
                yield _Plan(
                    plan.hexes + (hex_id,),
                    plan.nodes + (anchor,),
                    plan.length_m + step,
                    plan.rank_m + step * weight,
                    plan.visited | {hex_id},
                )

    def _prune(self, plans) -> list[_Plan]:
        best: dict = {}
        for plan in plans:
            key = (plan.visited, plan.nodes[-1])
            if key not in best or self._rank(plan) < self._rank(best[key]):
                best[key] = plan
        return sorted(best.values(), key=self._rank)[: self.beam_width]

    def new_hexes(self, hexes) -> int:
        """How many cells of a plan are new (not already yours)."""
        return sum(1 for hex_id in hexes if hex_id not in self.claimed)

    def plan_direction(self, first_hex: HexId) -> list[tuple[int, float, tuple]]:
        """Beam search whose first step goes to ``first_hex``.

        Returns up to ``MAX_BUILDS`` closed plans as (hexagons, length, nodes),
        best first: more hexagons, then closer to the target length. A plan is
        only closed once it is at least ``target_m - tolerance`` long; shorter
        plans must take another hexagon (or are dropped if none fits).
        """
        root = _Plan((self.start_hex,), (self.start_node,), 0.0, 0.0, frozenset({self.start_hex}))
        beam = self._prune(self._steps(root, [first_hex]))
        closed = []
        while beam:
            self.check_time()
            for plan in beam:
                total = plan.length_m + self.home[plan.nodes[-1]]
                if total >= self.low_m - _EPS:
                    closed.append((self.new_hexes(plan.hexes), total, plan.nodes))
            beam = self._prune(
                child for plan in beam for child in self._steps(plan, neighbours(plan.hexes[-1]))
            )
        closed.sort(key=lambda c: (-c[0], abs(c[1] - self.target_m)))
        return closed[:MAX_BUILDS]

    def build(self, nodes: tuple) -> list | None:
        """Join the plan's nodes, and finally the start, by shortest paths with the street-reuse penalty."""
        used: set[frozenset] = set()
        route = [nodes[0]]
        for source, target in zip(nodes, nodes[1:] + (nodes[0],)):
            try:
                leg = nx.shortest_path(
                    self.graph, source, target, weight=_penalised_weight(self.graph, used, self.penalty)
                )
            except nx.NetworkXNoPath:
                return None
            used.update(frozenset(edge) for edge in zip(leg, leg[1:]))
            route.extend(leg[1:])
        return route


def plan_route(
    graph: nx.Graph,
    start_node,
    target_m: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    min_depth_m: float = DEFAULT_MIN_DEPTH_M,
    claimed: frozenset = frozenset(),
    radius_m: float | None = None,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    cutoff_m: float = DEFAULT_CUTOFF_M,
    penalty: float = DEFAULT_PENALTY,
    time_limit_s: float | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> PlannedRoute:
    """Plan a closed loop hexagon by hexagon (see the module docstring).

    The beam search (``beam_width`` partial plans kept per step) is run once
    for each neighbour of the start cell as the first step. Each
    direction's best plan that becomes a street route within tolerance is
    scored; the best by safe hexagons, then least wasted distance, is polished
    and returned. Cells in ``claimed`` (already yours) are not counted as new and
    are avoided where possible. Only hexagons within ``radius_m`` of the start are planned
    (all hexagons with safe nodes if None). A step only looks ``cutoff_m`` of
    street distance ahead, so it stays fast.

    ``on_stage`` is called with STAGE_CHOOSING when the search for hexagons
    starts and with STAGE_JOINING when the chosen hexagons are joined by streets.

    Raises PlannerError if no plan gives a route within tolerance, and
    PlannerTimeout if ``time_limit_s`` runs out before polishing starts.
    """
    _validate_min_depth(min_depth_m)
    claimed = frozenset(claimed)
    if start_node not in graph:
        raise ValueError(f"start_node {start_node!r} is not in the graph")
    if target_m <= 0:
        raise ValueError(f"target_m must be positive, got {target_m}")
    if beam_width < 1:
        raise ValueError(f"beam_width must be at least 1, got {beam_width}")
    deadline = None if time_limit_s is None else time.perf_counter() + time_limit_s

    hex_ids = None
    if radius_m is not None:
        node = graph.nodes[start_node]
        hex_ids = hexes_within_radius(node["y"], node["x"], radius_m)

    if on_stage:
        on_stage(STAGE_CHOOSING)
    planner = _Planner(
        graph, start_node, target_m, tolerance, min_depth_m, hex_ids,
        beam_width, cutoff_m, penalty, deadline, claimed,
    )
    planner.check_time()

    plans = [(first_hex, planner.plan_direction(first_hex)) for first_hex in neighbours(planner.start_hex)]

    if on_stage:
        on_stage(STAGE_JOINING)
    best = None
    directions = 0
    for first_hex, closed in plans:
        for n_hexes, _, nodes in closed:
            planner.check_time()
            route = planner.build(nodes)
            if route is None:
                continue
            length = route_length(graph, route)
            if not planner.low_m - _EPS <= length <= planner.high_m + _EPS:
                continue
            score = score_route(graph, route, planner.pieces, min_depth_m, claimed)
            directions += 1
            key = (score.unique_hexes, -score.wasted_m)
            if best is None or key > best[0]:
                best = (key, route, first_hex, n_hexes)
            break
    if best is None:
        raise PlannerError(
            f"No hexagon plan gave a route within {tolerance:.0%} of {target_m:.0f} m"
        )

    planner.check_time()
    _, route, first_hex, n_hexes = best
    polished = polish_route(graph, route, target_m, tolerance, min_depth_m=min_depth_m, claimed=claimed)
    return PlannedRoute(
        route=polished.route,
        unique_hexes=polished.hexes_after,
        length_m=polished.length_after_m,
        wasted_m=polished.wasted_after_m,
        first_hex=first_hex,
        planned_hexes=n_hexes,
        directions_tried=directions,
        polished=polished,
    )


# --- planner versus random search ----------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    name: str  # "planner" or "random search"
    seconds: float
    route: list | None = None  # None if it failed
    safe_hexes: int = 0  # new cells, as drawn on the map: hexes_for_track with the safety margin, minus claimed
    length_m: float = 0.0
    wasted_m: float = 0.0
    polished: PolishedRoute | None = None
    error: str | None = None


@dataclass(frozen=True)
class RouteChoice:
    chosen: SearchResult
    planner: SearchResult
    random: SearchResult
    reason: str


def _result(name, graph, polished, seconds, min_depth_m, claimed) -> SearchResult:
    safe = hexes_for_track(route_to_latlons(graph, polished.route), min_depth_m) - claimed
    return SearchResult(
        name, seconds, polished.route, len(safe), polished.length_after_m,
        polished.wasted_after_m, polished,
    )


def _random_search(graph, start_node, target_m, tolerance, min_depth_m, n_candidates, seed, claimed) -> SearchResult:
    """``optimise_route`` + ``polish_route``; a failure is returned, not raised."""
    began = time.perf_counter()
    try:
        best = optimise_route(
            graph, start_node, target_m, n_candidates=n_candidates, seed=seed,
            tolerance=tolerance, min_depth_m=min_depth_m, claimed=claimed, verbose=False,
        )
        polished = polish_route(graph, best.route, target_m, tolerance, min_depth_m=min_depth_m, claimed=claimed)
        return _result("random search", graph, polished, time.perf_counter() - began, min_depth_m, claimed)
    except LoopNotFoundError as error:
        return SearchResult("random search", time.perf_counter() - began, error=str(error))


def choose_route(
    graph: nx.Graph,
    start_node,
    target_m: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    min_depth_m: float = DEFAULT_MIN_DEPTH_M,
    claimed: frozenset = frozenset(),
    n_candidates: int = 100,
    seed: int = 0,
    radius_m: float | None = None,
    time_limit_s: float | None = DEFAULT_TIME_LIMIT_S,
) -> RouteChoice:
    """Run the random search (``optimise_route`` + ``polish_route``) and the planner; keep the better.

    The one with more safe hexagons wins; a tie goes to less wasted distance,
    then to the random search. If the planner fails, or takes longer than
    ``time_limit_s`` in total, the random search result is used. Raises
    LoopNotFoundError if both fail.
    """
    claimed = frozenset(claimed)
    random = _random_search(graph, start_node, target_m, tolerance, min_depth_m, n_candidates, seed, claimed)

    began = time.perf_counter()
    try:
        planned = plan_route(
            graph, start_node, target_m, tolerance=tolerance,
            min_depth_m=min_depth_m, claimed=claimed, radius_m=radius_m, time_limit_s=time_limit_s,
        )
        seconds = time.perf_counter() - began
        if time_limit_s is not None and seconds > time_limit_s:
            raise PlannerTimeout(f"the planner took {seconds:.0f} s, over the {time_limit_s:g} s limit")
        planner = _result("planner", graph, planned.polished, seconds, min_depth_m, claimed)
    except Exception as error:  # any planner failure falls back to the random search
        message = str(error) if isinstance(error, PlannerError) else f"{type(error).__name__}: {error}"
        planner = SearchResult("planner", time.perf_counter() - began, error=message)

    if planner.route is None and random.route is None:
        raise LoopNotFoundError(f"random search: {random.error}; planner: {planner.error}")
    if planner.route is None:
        return RouteChoice(random, planner, random, f"planner failed ({planner.error})")
    if random.route is None:
        return RouteChoice(planner, planner, random, f"random search failed ({random.error})")

    if (planner.safe_hexes, -planner.wasted_m) > (random.safe_hexes, -random.wasted_m):
        chosen = planner
    else:
        chosen = random
    if planner.safe_hexes != random.safe_hexes:
        reason = "more safe hexagons"
    elif planner.wasted_m != random.wasted_m:
        reason = "same safe hexagons, less wasted distance"
    else:
        reason = "a tie, so the random search"
    return RouteChoice(chosen, planner, random, reason)


def choose_route_fast(
    graph: nx.Graph,
    start_node,
    target_m: float,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    min_depth_m: float = DEFAULT_MIN_DEPTH_M,
    claimed: frozenset = frozenset(),
    n_candidates: int = 100,
    seed: int = 0,
    radius_m: float | None = None,
    time_limit_s: float | None = DEFAULT_TIME_LIMIT_S,
    fallback_max_m: float = FALLBACK_MAX_M,
    on_stage: Callable[[str], None] | None = None,
) -> RouteChoice:
    """Planner only; the random search is skipped unless the planner fails.

    If the planner fails (or takes longer than ``time_limit_s``), the random
    search is used instead, but only for routes of at most ``fallback_max_m``
    (12 km). Longer routes, and a fallback that fails too, raise
    LoopNotFoundError. ``on_stage`` is told about the planner's stages and
    about STAGE_FALLBACK. The skipped search is in the result with
    ``error == SKIPPED``.
    """
    began = time.perf_counter()
    try:
        planned = plan_route(
            graph, start_node, target_m, tolerance=tolerance,
            min_depth_m=min_depth_m, claimed=claimed, radius_m=radius_m, time_limit_s=time_limit_s, on_stage=on_stage,
        )
        seconds = time.perf_counter() - began
        if time_limit_s is not None and seconds > time_limit_s:
            raise PlannerTimeout(f"the planner took {seconds:.0f} s, over the {time_limit_s:g} s limit")
        planner = _result("planner", graph, planned.polished, seconds, min_depth_m, claimed)
    except Exception as error:  # any planner failure may fall back to the random search
        message = str(error) if isinstance(error, PlannerError) else f"{type(error).__name__}: {error}"
        planner = SearchResult("planner", time.perf_counter() - began, error=message)

    skipped = SearchResult("random search", 0.0, error=SKIPPED)
    if planner.route is not None:
        return RouteChoice(planner, planner, skipped, "planner only")
    if target_m > fallback_max_m:
        raise LoopNotFoundError(
            f"planner: {planner.error}; the random search is only a fallback up to {fallback_max_m / 1000:g} km"
        )

    if on_stage:
        on_stage(STAGE_FALLBACK)
    claimed = frozenset(claimed)
    random = _random_search(graph, start_node, target_m, tolerance, min_depth_m, n_candidates, seed, claimed)
    if random.route is None:
        raise LoopNotFoundError(f"planner: {planner.error}; random search: {random.error}")
    return RouteChoice(random, planner, random, f"planner failed ({planner.error})")


def format_choice(choice: RouteChoice) -> str:
    """The two results side by side, and which one was chosen."""
    columns = [choice.planner, choice.random]

    def cells(result: SearchResult) -> list[str]:
        if result.error == SKIPPED:
            return ["skipped", "-", "-", "-"]
        if result.route is None:
            return ["failed", "-", "-", f"{result.seconds:.1f} s"]
        return [
            str(result.safe_hexes),
            f"{result.length_m / 1000:.2f} km",
            f"{result.wasted_m / 1000:.2f} km",
            f"{result.seconds:.1f} s",
        ]

    labels = ["safe hexagons", "length", "wasted distance", "time taken"]
    table = [cells(result) for result in columns]
    lines = [f"{'':<16}" + "".join(f"{result.name:>15}" for result in columns)]
    for i, label in enumerate(labels):
        lines.append(f"{label:<16}" + "".join(f"{column[i]:>15}" for column in table))
    lines.append(f"chosen: {choice.chosen.name} ({choice.reason})")
    return "\n".join(lines)
