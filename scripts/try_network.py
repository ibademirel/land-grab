"""Manual check: download the walking network around SW1A 1AA and print its size.

Needs internet access (postcodes.io and the OpenStreetMap Overpass API).
Run from the project root so the ./cache folder is shared with the app:

    .venv/bin/python scripts/try_network.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from land_grab.geo import postcode_to_latlon  # noqa: E402
from land_grab.network import get_network  # noqa: E402

POSTCODE = "SW1A 1AA"
RADIUS_M = 3000


def main() -> None:
    lat, lon = postcode_to_latlon(POSTCODE)
    print(f"{POSTCODE} -> lat {lat:.5f}, lon {lon:.5f}")
    graph = get_network(lat, lon, "run", RADIUS_M)
    print(f"nodes: {graph.number_of_nodes()}")
    print(f"edges: {graph.number_of_edges()}")


if __name__ == "__main__":
    main()
