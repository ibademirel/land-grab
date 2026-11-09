"""Manual check: optimise a 5 km loop from SW1A 1AA on the cached 3000 m run network.

Needs internet access the first time (postcodes.io and the OpenStreetMap
Overpass API); after that the network comes from ./cache. Run from the project
root so the ./cache folder is shared with the app:

    .venv/bin/python scripts/try_optimiser.py
"""

import statistics
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import osmnx as ox  # noqa: E402

from land_grab import optimiser  # noqa: E402
from land_grab.geo import postcode_to_latlon  # noqa: E402
from land_grab.network import get_network  # noqa: E402

POSTCODE = "SW1A 1AA"
RADIUS_M = 3000
TARGET_M = 5000
N_CANDIDATES = 100
SEED = 0


def main() -> None:
    lat, lon = postcode_to_latlon(POSTCODE)
    print(f"{POSTCODE} -> lat {lat:.5f}, lon {lon:.5f}")

    graph = get_network(lat, lon, "run", RADIUS_M)
    start_node = ox.distance.nearest_nodes(graph, X=lon, Y=lat)
    print(f"start node: {start_node}")

    # optimise_route returns only the best loop, so record every candidate's
    # score as it is computed. Only loops within tolerance are scored, and
    # identical loops from different seeds are scored once.
    scores = []
    real_score_route = optimiser.score_route

    def recording_score_route(*args, **kwargs):
        score = real_score_route(*args, **kwargs)
        scores.append(score.unique_hexes)
        return score

    with mock.patch.object(optimiser, "score_route", recording_score_route):
        best = optimiser.optimise_route(
            graph, start_node, TARGET_M, n_candidates=N_CANDIDATES, seed=SEED
        )

    print(f"best score: {best.unique_hexes} hexagons")
    print(f"median score: {statistics.median(scores):g} hexagons")
    print(f"lowest score: {min(scores)} hexagons")
    print(f"({len(scores)} distinct candidates scored)")


if __name__ == "__main__":
    main()
