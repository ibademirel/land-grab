"""Draw a route and the H3 cells it covers on an interactive folium map."""

import math
from pathlib import Path

import folium
import networkx as nx

from land_grab.hexgrid import HexId, hex_boundary, surrounding_hexes
from land_grab.optimiser import _edge_data

LatLon = tuple[float, float]

COVERED_COLOUR = "#e8590c"
ROUTE_COLOUR = "#1c7ed6"
OUTLINE_COLOUR = "#868e96"
RISKY_COLOUR = "#fcc419"
RISKY_TOOLTIP = "risky: route only clips this hexagon"
CLAIMED_COLOUR = "#7048e8"
CLAIMED_TOOLTIP = "already yours, not counted as new"


def route_to_latlons(graph: nx.Graph, route: list) -> list[LatLon]:
    """Return the (lat, lon) points along a route of graph nodes.

    Follows each edge's ``geometry`` where it has one (so curved streets are
    drawn as curves), otherwise the straight line between its nodes.
    """
    if not route:
        return []
    first = graph.nodes[route[0]]
    points: list[LatLon] = [(first["y"], first["x"])]
    for u, v in zip(route, route[1:]):
        geometry = _edge_data(graph, u, v).get("geometry")
        if geometry is not None:
            edge_points = [(lat, lon) for lon, lat in geometry.coords]
            start = (graph.nodes[u]["y"], graph.nodes[u]["x"])
            if math.dist(edge_points[-1], start) < math.dist(edge_points[0], start):
                edge_points.reverse()
        else:
            edge_points = [(graph.nodes[v]["y"], graph.nodes[v]["x"])]
            points.append(edge_points[0])
            continue
        points.extend(edge_points[1:])
    return points


def _polygon_locations(hex_id: HexId) -> list[LatLon]:
    """The cell's corners as (lat, lon), straight from H3 (folium wants lat first)."""
    return hex_boundary(hex_id)


def draw_map(
    start_latlon: LatLon,
    route_latlons: list[LatLon],
    hex_ids: set[HexId],
    out_path: str | Path,
    *,
    margin_rings: int = 2,
    risky_hex_ids: set[HexId] | None = None,
    claimed_hex_ids: set[HexId] | None = None,
) -> Path:
    """Save an HTML map of a route and the hexagons it covers, and return its path.

    - The route is drawn as a line.
    - Hexagons in ``hex_ids`` are filled; the hexagons within ``margin_rings``
      steps of them (the rest of the area) get a faint outline only.
    - Hexagons in ``risky_hex_ids`` (touched by the route but not far enough
      inside to count, see ``risky_hexes``) are drawn in a lighter colour with a
      dashed outline and a hover text saying the route only clips them.
    - Hexagons in ``claimed_hex_ids`` (cells you already own) are drawn in
      purple with a hover text saying they are not counted as new.
    - The start point is marked with a green marker.

    The map is zoomed to fit the hexagons and the route. Missing parent
    folders of ``out_path`` are created.
    """
    if margin_rings < 0:
        raise ValueError(f"margin_rings must not be negative, got {margin_rings}")

    fmap = folium.Map(location=start_latlon, zoom_start=14, tiles="OpenStreetMap")

    claimed = set(claimed_hex_ids or ())
    risky = set(risky_hex_ids or ()) - set(hex_ids) - claimed
    hex_ids = set(hex_ids) - claimed
    area = surrounding_hexes(hex_ids | risky, margin_rings)

    for hex_id in sorted(area - hex_ids - risky - claimed):
        folium.Polygon(
            locations=_polygon_locations(hex_id),
            color=OUTLINE_COLOUR,
            weight=1,
            opacity=0.35,
            fill=False,
        ).add_to(fmap)

    for hex_id in sorted(risky):
        folium.Polygon(
            locations=_polygon_locations(hex_id),
            color=RISKY_COLOUR,
            weight=2,
            dash_array="6 6",
            fill=True,
            fill_color=RISKY_COLOUR,
            fill_opacity=0.15,
            tooltip=f"{hex_id} - {RISKY_TOOLTIP}",
        ).add_to(fmap)

    for hex_id in sorted(claimed):
        folium.Polygon(
            locations=_polygon_locations(hex_id),
            color=CLAIMED_COLOUR,
            weight=2,
            fill=True,
            fill_color=CLAIMED_COLOUR,
            fill_opacity=0.3,
            tooltip=f"{hex_id} - {CLAIMED_TOOLTIP}",
        ).add_to(fmap)

    for hex_id in sorted(hex_ids):
        folium.Polygon(
            locations=_polygon_locations(hex_id),
            color=COVERED_COLOUR,
            weight=2,
            fill=True,
            fill_color=COVERED_COLOUR,
            fill_opacity=0.35,
            tooltip=hex_id,
        ).add_to(fmap)

    if route_latlons:
        folium.PolyLine(route_latlons, color=ROUTE_COLOUR, weight=4, opacity=0.9).add_to(fmap)

    folium.Marker(
        start_latlon, tooltip="Start", icon=folium.Icon(color="green", icon="play")
    ).add_to(fmap)

    corners = [start_latlon, *route_latlons]
    # Claimed cells far from the route do not stretch the view; those near it do.
    for hex_id in hex_ids | risky | (claimed & area):
        corners.extend(_polygon_locations(hex_id))
    lats, lons = [p[0] for p in corners], [p[1] for p in corners]
    fmap.fit_bounds([(min(lats), min(lons)), (max(lats), max(lons))])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    return out_path
