import networkx as nx
import pytest
from shapely.geometry import LineString

from land_grab.hexgrid import _TO_WGS84, RESOLUTION, hex_centre_bng, latlon_to_hex, neighbours
from land_grab.scoring import annotate_edges

A, B, C = 1, 2, 3

# Three cells in a row: HOME, then NEXT across one of its edges, then LAST across the far side of NEXT.
HOME = latlon_to_hex(51.5010, -0.1416)
NEXT = neighbours(HOME)[0]
_E0, _N0 = hex_centre_bng(HOME)
_E1, _N1 = hex_centre_bng(NEXT)


def _latlon(de, dn):
    lon, lat = _TO_WGS84.transform(_E0 + de, _N0 + dn)
    return lat, lon


# B is the centre of NEXT (so A-B crosses from HOME into NEXT); C is a little way from B.
_BE, _BN = _E1 - _E0, _N1 - _N0


def _tiny_graph():
    """Three nodes, two edges.

    A (centre of HOME) --straight--> B (centre of NEXT)
    B --geometry--> C (200 m from B, same cell), but the geometry detours back to
    the centre of HOME and returns, so it also touches HOME. A straight line B-C
    would stay inside NEXT.
    """
    g = nx.MultiDiGraph()
    for node, (de, dn) in {A: (0, 0), B: (_BE, _BN), C: (_BE + 200, _BN)}.items():
        lat, lon = _latlon(de, dn)
        g.add_node(node, y=lat, x=lon)

    g.add_edge(A, B)  # no geometry: straight line between its nodes
    detour = [_latlon(_BE, _BN), _latlon(0, 0), _latlon(_BE + 200, _BN)]
    g.add_edge(B, C, geometry=LineString([(lon, lat) for lat, lon in detour]))
    return g


def test_hex_sets_for_straight_edge_and_geometry_edge():
    g = annotate_edges(_tiny_graph(), _latlon(0, 0))

    assert {HOME, NEXT} <= g.edges[A, B, 0]["hexes"]
    assert {HOME, NEXT} <= g.edges[B, C, 0]["hexes"]
    assert all(isinstance(h, str) for h in g.edges[A, B, 0]["hexes"])


def test_geometry_is_followed_not_the_straight_line():
    straight = _tiny_graph()
    straight.edges[B, C, 0].pop("geometry")
    straight = annotate_edges(straight, _latlon(0, 0))
    assert straight.edges[B, C, 0]["hexes"] == {NEXT}


def test_returns_the_same_graph_object():
    g = _tiny_graph()
    assert annotate_edges(g, _latlon(0, 0)) is g


def test_graph_level_lookup_maps_cell_to_edges():
    g = annotate_edges(_tiny_graph(), _latlon(0, 0))

    assert g.graph["hex_resolution"] == RESOLUTION
    lookup = g.graph["hex_edges"]
    assert (A, B, 0) in lookup[HOME] and (B, C, 0) in lookup[HOME]
    assert (A, B, 0) in lookup[NEXT] and (B, C, 0) in lookup[NEXT]
    for cell, edges in lookup.items():
        assert all(cell in g.edges[e]["hexes"] for e in edges)


def test_edge_outside_radius_gets_empty_hexes_and_is_not_in_lookup():
    far = 5000
    g = _tiny_graph()
    lat, lon = _latlon(far, far)
    g.add_node(9, y=lat, x=lon)
    g.add_edge(C, 9)
    g = annotate_edges(g, _latlon(0, 0), radius_m=100)

    assert g.edges[A, B, 0]["hexes"]
    assert g.edges[C, 9, 0]["hexes"] == set()
    assert (C, 9, 0) not in {e for edges in g.graph["hex_edges"].values() for e in edges}


def test_radius_uses_distance_to_the_edge_not_just_its_nodes():
    # Both B and C are far from the origin, outside the radius, but the B-C geometry passes through
    # the origin, so the edge counts as inside.
    g = annotate_edges(_tiny_graph(), _latlon(0, 0), radius_m=300)

    assert g.edges[B, C, 0]["hexes"]


def test_every_edge_gets_a_hexes_attribute():
    g = annotate_edges(_tiny_graph(), _latlon(0, 0), radius_m=1)
    assert all("hexes" in data for _, _, data in g.edges(data=True))


def test_annotating_twice_rebuilds_the_lookup():
    g = _tiny_graph()
    annotate_edges(g, _latlon(0, 0))
    lat, lon = _latlon(50_000, 0)
    annotate_edges(g, (lat, lon), radius_m=100)
    assert g.graph["hex_edges"] == {}


def test_invalid_arguments_raise_value_error():
    with pytest.raises(ValueError):
        annotate_edges(_tiny_graph(), _latlon(0, 0), radius_m=-1)
