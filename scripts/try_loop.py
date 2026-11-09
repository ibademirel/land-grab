"""Manual check: build a 5 km loop from SW1A 1AA on the cached 3000 m run network.

Needs internet access the first time (postcodes.io and the OpenStreetMap
Overpass API); after that the network comes from ./cache. Run from the project
root so the ./cache folder is shared with the app:

    .venv/bin/python scripts/try_loop.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import osmnx as ox  # noqa: E402

from land_grab.geo import postcode_to_latlon  # noqa: E402
from land_grab.network import get_network  # noqa: E402
from land_grab.router import make_loop, route_length  # noqa: E402

POSTCODE = "SW1A 1AA"
RADIUS_M = 3000
TARGET_M = 5000
SEED = 1


def return_reuse_fraction(graph, route) -> float:
    """Fraction of the return half's streets that were already used on the way out.

    The route is split at the first node where its real length passes half the
    total. Streets are compared without direction.
    """
    half = route_length(graph, route) / 2
    walked, split = 0.0, 0
    while walked < half:
        walked += route_length(graph, route[split : split + 2])
        split += 1
    outward = {frozenset(edge) for edge in zip(route[: split + 1], route[1 : split + 1])}
    back = [frozenset(edge) for edge in zip(route[split:], route[split + 1 :])]
    return sum(street in outward for street in back) / len(back)


def main() -> None:
    lat, lon = postcode_to_latlon(POSTCODE)
    print(f"{POSTCODE} -> lat {lat:.5f}, lon {lon:.5f}")

    graph = get_network(lat, lon, "run", RADIUS_M)
    start_node = ox.distance.nearest_nodes(graph, X=lon, Y=lat)
    print(f"start node: {start_node}")

    loop = make_loop(graph, start_node, TARGET_M, seed=SEED)
    print(f"nodes in loop: {len(loop)}")
    print(f"real length: {route_length(graph, loop):.0f} m (target {TARGET_M} m)")
    print(f"return half reusing outward streets: {return_reuse_fraction(graph, loop):.1%}")


if __name__ == "__main__":
    main()
