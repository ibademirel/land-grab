"""Annotate street-network edges with the hexagons they pass through."""

import networkx as nx
from shapely.geometry import LineString, Point

from land_grab.coverage import hexes_for_track
from land_grab.hexgrid import _TO_BNG, RESOLUTION, HexId


def _edge_latlon_points(
    graph: nx.MultiDiGraph, u, v, data: dict
) -> list[tuple[float, float]]:
    """The edge's (lat, lon) points: its geometry, or the straight line between its nodes."""
    geometry = data.get("geometry")
    if geometry is not None:
        return [(lat, lon) for lon, lat in geometry.coords]
    return [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in (u, v)]


def _distance_to_origin_m(points: list[tuple[float, float]], origin: Point) -> float:
    line = LineString([_TO_BNG.transform(lon, lat) for lat, lon in points])
    return line.distance(origin)


def annotate_edges(
    graph: nx.MultiDiGraph,
    origin_latlon: tuple[float, float],
    radius_m: float | None = None,
) -> nx.MultiDiGraph:
    """Add the hexagons each edge passes through, modifying the graph in place.

    Every edge gets a ``"hexes"`` attribute: the set of H3 cell ids (see
    ``land_grab.hexgrid``) its geometry touches. The geometry is the edge's
    ``"geometry"`` if it has one, otherwise the straight line between its nodes.
    Hexes are found by densifying the line to 20 m steps (``hexes_for_track``).

    If ``radius_m`` is given, only edges that come within ``radius_m`` metres
    of ``origin_latlon`` are annotated. Edges entirely outside get an empty
    ``"hexes"`` set and are left out of the lookup below. An edge counts as
    inside if any part of it is in range, even when both its nodes are outside.

    The graph-level attribute ``graph.graph["hex_edges"]`` maps each hex id to
    the set of ``(u, v, key)`` edges that pass through it, and
    ``graph.graph["hex_resolution"]`` records the H3 resolution used. Both are
    rebuilt from scratch on every call. Returns the same graph.
    """
    if radius_m is not None and radius_m < 0:
        raise ValueError(f"radius_m must not be negative, got {radius_m}")

    origin_lat, origin_lon = origin_latlon
    origin = Point(*_TO_BNG.transform(origin_lon, origin_lat))

    hex_edges: dict[HexId, set[tuple]] = {}
    for u, v, key, data in graph.edges(keys=True, data=True):
        points = _edge_latlon_points(graph, u, v, data)
        if radius_m is not None and _distance_to_origin_m(points, origin) > radius_m:
            data["hexes"] = set()
            continue
        hexes = hexes_for_track(points, min_depth_m=0)  # every hexagon the edge enters
        data["hexes"] = hexes
        for hex_id in hexes:
            hex_edges.setdefault(hex_id, set()).add((u, v, key))

    graph.graph["hex_resolution"] = RESOLUTION
    graph.graph["hex_edges"] = hex_edges
    return graph
