"""Polish a loop: remove spurs, zigzags and detours, then spend spare distance on extra hexagons."""

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from land_grab.coverage import DEFAULT_MIN_DEPTH_M, _meets_margin, _validate_min_depth
from land_grab.hexgrid import _TO_BNG, HexId, bng_to_hex, depth_in_hex, neighbours
from land_grab.optimiser import RouteScore, _edge_pieces, score_route
from land_grab.router import DEFAULT_TOLERANCE, _edge_length, route_length

DEFAULT_MAX_ROUNDS = 50
DEFAULT_QUIRK_MAX_M = 150.0  # longest spur or zigzag (as walked) that counts as a quirk
DEFAULT_MAX_STRETCH_M = 2000.0  # longest stretch of route tried for a shortcut
NEAREST_ROUTE_NODES = 5  # route nodes closest to a waypoint that insertions are tried next to

_EPS = 1e-6


@dataclass(frozen=True)
class RouteQuirk:
    """A small unnecessary bit of a route.

    ``kind`` is "spur" (A -> B -> A along the same street, possibly several
    streets deep) or "zigzag" (a stretch that a shorter path between its two
    ends could replace). ``start``/``end`` are the first and last indices of the
    stretch in the route (both inclusive; a spur starts and ends on the same
    node). ``length_m`` is the stretch as walked, ``saving_m`` how much shorter
    the route gets without it, and ``shortcut`` the nodes that replace the
    stretch (a legal path in the graph; just the junction for a spur).
    ``latlon`` is the approximate position: the tip of a spur, the middle of a zigzag.
    """

    kind: str
    start: int
    end: int
    length_m: float
    saving_m: float
    latlon: tuple[float, float]
    shortcut: tuple


@dataclass(frozen=True)
class KeptQuirk:
    """A quirk that is still in the polished route, and why it was left alone."""

    quirk: RouteQuirk
    reason: str
    hexes: tuple = ()  # the safe hexagons that removing it would lose


@dataclass(frozen=True)
class PolishedRoute:
    route: list  # nodes, start to start
    hexes_before: int
    hexes_after: int
    length_before_m: float  # real length, never the street-reuse penalised one
    length_after_m: float
    wasted_before_m: float
    wasted_after_m: float
    rounds: int  # rounds that improved the route
    quirks_before: list = field(default_factory=list)  # RouteQuirk, in the route as given
    quirks_kept: list = field(default_factory=list)  # KeptQuirk, still in the polished route


def _node_latlon(graph: nx.Graph, node) -> tuple[float, float]:
    return graph.nodes[node]["y"], graph.nodes[node]["x"]


def find_route_quirks(
    graph: nx.Graph, route: list, max_length_m: float = DEFAULT_QUIRK_MAX_M
) -> list[RouteQuirk]:
    """List the spurs and zigzags in a route, in route order.

    - Spur: the route goes A -> B -> A along the same street (or further:
      A -> B -> C -> B -> A). Reported if it is shorter than ``max_length_m``
      as walked, there and back. A route that is one big out-and-back has no spur.
    - Zigzag: a stretch shorter than ``max_length_m`` that visits each node once
      and that a shorter path between its two ends could replace. The shorter
      path is searched in ``graph`` itself (respecting one-way edges), so it only
      uses streets, pavements and crossings the graph allows: a zigzag that is a
      real road crossing, with no shorter legal way in the graph, is not reported.
    """
    n = len(route)
    cum = [0.0]
    for u, v in zip(route, route[1:]):
        cum.append(cum[-1] + _edge_length(graph, u, v))

    quirks: list[RouteQuirk] = []

    i = 1
    while i < n - 1:
        if route[i - 1] != route[i + 1]:
            i += 1
            continue
        k = 1
        while i - k - 1 >= 0 and i + k + 1 < n and route[i - k - 1] == route[i + k + 1]:
            k += 1
        while k > 0 and cum[i + k] - cum[i - k] >= max_length_m:
            k -= 1
        if k == 0 or (i - k == 0 and i + k == n - 1):
            i += 1
            continue
        length = cum[i + k] - cum[i - k]
        quirks.append(
            RouteQuirk(
                "spur", i - k, i + k, length, length, _node_latlon(graph, route[i]), (route[i - k],)
            )
        )
        i += k + 1

    i = 0
    while i < n - 2:
        last = i
        seen = {route[i]}
        for j in range(i + 1, n):
            if cum[j] - cum[i] >= max_length_m or route[j] in seen:
                break
            seen.add(route[j])
            last = j
        if last < i + 2:
            i += 1
            continue
        dist, paths = nx.single_source_dijkstra(
            graph, route[i], cutoff=cum[last] - cum[i], weight="length"
        )
        best = None
        for j in range(i + 2, last + 1):
            walked = cum[j] - cum[i]
            shortest = dist.get(route[j])
            if shortest is not None and walked - shortest > _EPS:
                if best is None or walked - shortest > best[0] + _EPS:
                    best = (walked - shortest, j)
        if best is None:
            i += 1
            continue
        saving, j = best
        quirks.append(
            RouteQuirk(
                "zigzag",
                i,
                j,
                cum[j] - cum[i],
                saving,
                _node_latlon(graph, route[(i + j) // 2]),
                tuple(paths[route[j]]),
            )
        )
        i = j
    quirks.sort(key=lambda q: (q.start, q.end))
    return quirks


def _without_quirk(route: list, quirk: RouteQuirk) -> list:
    return route[: quirk.start] + list(quirk.shortcut) + route[quirk.end + 1 :]


class _Polisher:
    """The two improvement steps, sharing one cache of edge pieces and shortest paths."""

    def __init__(self, graph, low_m, high_m, max_stretch_m, min_depth_m, quirk_max_m, claimed):
        self.graph = graph
        self.claimed = claimed
        self.low_m = low_m
        self.high_m = high_m
        self.max_stretch_m = max_stretch_m
        self.min_depth_m = min_depth_m
        self.quirk_max_m = quirk_max_m
        self._pieces: dict = {}
        self._paths: dict = {}
        self._xy: dict | None = None
        self._nodes_by_hex: dict[HexId, list] | None = None

    # --- helpers -------------------------------------------------------------

    def score(self, route: list) -> RouteScore:
        return score_route(self.graph, route, self._pieces, self.min_depth_m, self.claimed)

    def hexes(self, route: list) -> set[HexId]:
        """New cells the route properly covers (clipped ones are not covered, so they can be deepened).

        Cells you already own (``claimed``) are left out: they are not new points.
        """
        covered: set[HexId] = set()
        for u, v in zip(route, route[1:]):
            if (u, v) not in self._pieces:
                self._pieces[(u, v)] = _edge_pieces(self.graph, u, v)
            covered.update(
                hex_id
                for hex_id, _, depth_m in self._pieces[(u, v)]
                if hex_id not in self.claimed and _meets_margin(depth_m, self.min_depth_m)
            )
        if len(route) == 1:
            x, y = self.node_xy(route[0])
            hex_id = bng_to_hex(x, y)
            if hex_id not in self.claimed and _meets_margin(depth_in_hex(x, y, hex_id), self.min_depth_m):
                covered.add(hex_id)
        return covered

    def excess(self, length_m: float) -> float:
        """How far a length is outside the allowed band (0 inside it)."""
        return max(0.0, self.low_m - length_m, length_m - self.high_m)

    def shortest_path(self, source, target) -> list | None:
        key = (source, target)
        if key not in self._paths:
            try:
                self._paths[key] = nx.shortest_path(self.graph, source, target, weight="length")
            except nx.NetworkXNoPath:
                self._paths[key] = None
        return self._paths[key]

    def _load_nodes(self) -> None:
        nodes = list(self.graph.nodes)
        lons = [self.graph.nodes[n]["x"] for n in nodes]
        lats = [self.graph.nodes[n]["y"] for n in nodes]
        eastings, northings = _TO_BNG.transform(lons, lats)
        self._xy = dict(zip(nodes, zip(eastings, northings)))
        self._nodes_by_hex = {}
        for node, (x, y) in self._xy.items():
            self._nodes_by_hex.setdefault(bng_to_hex(x, y), []).append(node)

    def node_xy(self, node) -> tuple[float, float]:
        if self._xy is None:
            self._load_nodes()
        return self._xy[node]

    def nodes_in_hex(self, hex_id: HexId) -> list:
        if self._nodes_by_hex is None:
            self._load_nodes()
        return self._nodes_by_hex.get(hex_id, [])

    # --- step 0: remove spurs and zigzags --------------------------------------

    def _removal_blocker(self, route: list, quirk: RouteQuirk):
        """Return (reason, lost_hexes) if the quirk must stay, or None if it can go."""
        new_route = _without_quirk(route, quirk)
        lost = self.hexes(route) - self.hexes(new_route)
        if lost:
            return "earns a safe hexagon", tuple(sorted(lost))
        length = route_length(self.graph, route)
        new_length = route_length(self.graph, new_route)
        if self.excess(new_length) > self.excess(length) + _EPS:
            return "needed to stay within the length limit", ()
        return None

    def remove_quirks(self, route: list) -> tuple[list, bool]:
        """Remove spurs and zigzags that are not needed.

        A quirk goes unless removing it would lose a safe hexagon (one that meets
        ``min_depth_m``) or push the length further outside the allowed band.
        """
        route = list(route)
        changed = False
        kept: set = set()
        while True:
            for quirk in find_route_quirks(self.graph, route, self.quirk_max_m):
                key = (quirk.kind, tuple(route[quirk.start : quirk.end + 1]))
                if key in kept:
                    continue
                if self._removal_blocker(route, quirk) is None:
                    route, changed = _without_quirk(route, quirk), True
                    break
                kept.add(key)
            else:
                return route, changed

    def kept_quirks(self, route: list) -> list[KeptQuirk]:
        """The quirks still in the route, each with the reason it was not removed."""
        kept = []
        for quirk in find_route_quirks(self.graph, route, self.quirk_max_m):
            blocker = self._removal_blocker(route, quirk)
            reason, lost = blocker if blocker else ("left alone", ())
            kept.append(KeptQuirk(quirk, reason, lost))
        return kept

    # --- step 1: remove detours ----------------------------------------------

    def remove_detours(self, route: list) -> tuple[list, bool]:
        """Replace stretches of the route by shortest paths where that is shorter.

        A change is kept if the route has at least as many unique hexagons and
        is shorter, and is no further outside the allowed length band than before.
        """
        graph = self.graph
        route = list(route)
        changed = False
        length = route_length(graph, route)
        hexes = self.score(route).unique_hexes

        i = 0
        while i < len(route) - 1:
            cum = [0.0]
            for u, v in zip(route, route[1:]):
                cum.append(cum[-1] + _edge_length(graph, u, v))
            last = i + 1
            while last + 1 < len(route) and cum[last + 1] - cum[i] <= self.max_stretch_m:
                last += 1
            dist, paths = nx.single_source_dijkstra(
                graph, route[i], cutoff=cum[last] - cum[i], weight="length"
            )

            options = []
            for j in range(i + 1, last + 1):
                shortest = dist.get(route[j])
                if shortest is not None and cum[j] - cum[i] - shortest > _EPS:
                    options.append((cum[j] - cum[i] - shortest, j))
            options.sort(reverse=True)

            for saving, j in options:
                if self.excess(length - saving) > self.excess(length) + _EPS:
                    continue
                new_route = route[:i] + paths[route[j]] + route[j + 1 :]
                if len(new_route) < 2:  # never collapse the loop to a single node
                    continue
                if self.score(new_route).unique_hexes >= hexes:
                    route, changed = new_route, True
                    length = route_length(graph, route)
                    break
            else:
                i += 1
        return route, changed

    # --- step 2: spend spare distance ----------------------------------------

    def spend_spare_distance(self, route: list) -> tuple[list, bool]:
        """Add hexagons next to the route by routing through a waypoint inside each.

        A change is kept only if the route gains hexagons and its real length
        stays inside the allowed band.
        """
        route = list(route)
        changed = False
        covered = self.hexes(route)

        # One waypoint per uncovered neighbouring hexagon: its node nearest the route.
        route_xy = np.array([self.node_xy(n) for n in route])
        targets = []
        seen: set[HexId] = set()
        for hex_id in sorted(covered):
            for neighbour in neighbours(hex_id):
                if neighbour in covered or neighbour in seen or neighbour in self.claimed:
                    continue
                seen.add(neighbour)
                nodes = self.nodes_in_hex(neighbour)
                if not nodes:
                    continue
                points = np.array([self.node_xy(n) for n in nodes])
                gaps = np.linalg.norm(points[:, None, :] - route_xy[None, :, :], axis=2)
                w, _ = np.unravel_index(np.argmin(gaps), gaps.shape)
                targets.append((float(gaps.min()), neighbour, nodes[w]))
        targets.sort(key=lambda t: t[0])

        for _, hex_id, waypoint in targets:
            if hex_id in covered:  # picked up by an earlier insertion
                continue
            new_route = self._insert_waypoint(route, waypoint)
            if new_route is not None:
                route, changed = new_route, True
                covered = self.hexes(route)
        return route, changed

    def _insert_waypoint(self, route: list, waypoint) -> list | None:
        """Route through ``waypoint`` between two consecutive route nodes, if that gains hexagons."""
        graph = self.graph
        if waypoint in route:
            return None
        length = route_length(graph, route)
        hexes = self.score(route).unique_hexes

        wx, wy = self.node_xy(waypoint)
        gaps = np.array([np.hypot(wx - x, wy - y) for x, y in (self.node_xy(n) for n in route)])
        segments = set()
        for k in np.argsort(gaps)[:NEAREST_ROUTE_NODES]:
            segments.update(s for s in (int(k) - 1, int(k)) if 0 <= s < len(route) - 1)

        options = []
        for i in segments:
            before, after = route[i], route[i + 1]
            to_waypoint = self.shortest_path(before, waypoint)
            from_waypoint = self.shortest_path(waypoint, after)
            if to_waypoint is None or from_waypoint is None:
                continue
            new_route = route[: i + 1] + to_waypoint[1:] + from_waypoint[1:] + route[i + 2 :]
            new_length = route_length(graph, new_route)
            if self.low_m - _EPS <= new_length <= self.high_m + _EPS:
                options.append((new_length, i, new_route))

        for _, _, new_route in sorted(options, key=lambda o: o[:2]):
            if self.score(new_route).unique_hexes > hexes:
                return new_route
        return None


def polish_route(
    graph: nx.Graph,
    route: list,
    target_m: float,
    tolerance: float = DEFAULT_TOLERANCE,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    *,
    max_stretch_m: float = DEFAULT_MAX_STRETCH_M,
    min_depth_m: float = DEFAULT_MIN_DEPTH_M,
    quirk_max_m: float = DEFAULT_QUIRK_MAX_M,
    claimed: frozenset = frozenset(),
) -> PolishedRoute:
    """Improve a closed loop (e.g. from ``optimise_route``) without changing its start.

    Each round has three steps, and rounds repeat until one makes no improvement
    or ``max_rounds`` is reached:

    0. Remove spurs and zigzags (see ``find_route_quirks``): short out-and-back
       stubs and little zigzags at junctions, up to ``quirk_max_m`` long. One is
       removed unless that would lose a safe hexagon, i.e. the spur is what earns
       it, or would push the length further outside the allowed band. The
       zigzag's replacement is a path in ``graph``, so it is always legal to walk.
       Quirks left in the result are listed in ``quirks_kept`` with the reason.
    1. Remove detours: a stretch of the route (up to ``max_stretch_m`` long) is
       replaced by the shortest path between its ends. Kept if the number of
       unique hexagons does not go down and the route gets shorter. A change is
       also refused if it moves the length further outside
       ``target_m`` +/- ``tolerance`` than it was.
    2. Spend spare distance: for each uncovered hexagon next to the route, take
       its graph node nearest the route and try routing through it, by shortest
       paths, between two consecutive route nodes. Kept only if the number of
       unique hexagons goes up and the length stays within
       ``target_m`` +/- ``tolerance``.

    Hexagons count only if the route gets ``min_depth_m`` inside them (the GPS
    safety margin, as in ``score_route``). Cells in ``claimed`` are not new points:
    they are not counted and are not worth a detour.

    Lengths are always the real street lengths, never the penalised ones used
    when building the loop. The route stays closed with the same start and end node.
    """
    _validate_min_depth(min_depth_m)
    if not route:
        raise ValueError("route must not be empty")
    if route[0] != route[-1]:
        raise ValueError("route must be closed: it must start and end at the same node")
    if target_m <= 0:
        raise ValueError(f"target_m must be positive, got {target_m}")
    if tolerance < 0:
        raise ValueError(f"tolerance must not be negative, got {tolerance}")
    if quirk_max_m < 0:
        raise ValueError(f"quirk_max_m must not be negative, got {quirk_max_m}")
    if max_rounds < 0:
        raise ValueError(f"max_rounds must not be negative, got {max_rounds}")

    polisher = _Polisher(
        graph,
        low_m=target_m * (1 - tolerance),
        high_m=target_m * (1 + tolerance),
        max_stretch_m=max_stretch_m,
        min_depth_m=min_depth_m,
        quirk_max_m=quirk_max_m,
        claimed=claimed,
    )
    before = polisher.score(route)
    quirks_before = find_route_quirks(graph, route, quirk_max_m)
    length_before = route_length(graph, route)

    current = list(route)
    rounds = 0
    for _ in range(max_rounds):
        current, tidied = polisher.remove_quirks(current)
        current, shortened = polisher.remove_detours(current)
        current, extended = polisher.spend_spare_distance(current)
        if not (tidied or shortened or extended):
            break
        rounds += 1

    after = polisher.score(current)
    return PolishedRoute(
        route=current,
        hexes_before=before.unique_hexes,
        hexes_after=after.unique_hexes,
        length_before_m=length_before,
        length_after_m=route_length(graph, current),
        wasted_before_m=before.wasted_m,
        wasted_after_m=after.wasted_m,
        rounds=rounds,
        quirks_before=quirks_before,
        quirks_kept=polisher.kept_quirks(current),
    )
