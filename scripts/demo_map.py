"""Full pipeline demo: a 10 km run from SW1A 1AA, saved as an HTML map.

Postcode -> street network -> planner and random search (the better one is kept) -> demo_map.html.

Add --gpx to also save the route as a GPX track (default demo_route.gpx, or
give a path):

    .venv/bin/python scripts/demo_map.py --gpx
    .venv/bin/python scripts/demo_map.py --gpx my_route.gpx

Needs internet access the first time (postcodes.io and the OpenStreetMap
Overpass API); after that the network comes from ./cache. Run from the project
root so the ./cache folder is shared with the app:

    .venv/bin/python scripts/demo_map.py
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import osmnx as ox  # noqa: E402

from land_grab.coverage import hexes_for_track, risky_hexes  # noqa: E402
from land_grab.export import save_gpx  # noqa: E402
from land_grab.geo import postcode_to_latlon  # noqa: E402
from land_grab.mapview import draw_map, route_to_latlons  # noqa: E402
from land_grab.network import get_network  # noqa: E402
from land_grab.planner import choose_route, format_choice  # noqa: E402

POSTCODE = "SW1A 1AA"
ACTIVITY = "run"
TARGET_M = 10_000
RADIUS_M = 3000  # network area; a 10 km loop fits inside it
N_CANDIDATES = 100
SEED = 0
MIN_DEPTH_M = 25  # GPS safety margin: a hexagon counts only if the route gets this far inside it
OUT_PATH = PROJECT_ROOT / "demo_map.html"
GPX_PATH = PROJECT_ROOT / "demo_route.gpx"


def describe_quirk(quirk) -> str:
    lat, lon = quirk.latlon
    return (
        f"{quirk.kind}: {quirk.length_m:.0f} m walked, saves {quirk.saving_m:.0f} m, "
        f"near {lat:.5f}, {lon:.5f}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a hexagon-claiming loop and draw it on a map.")
    parser.add_argument(
        "--gpx",
        nargs="?",
        const=GPX_PATH,
        default=None,
        type=Path,
        metavar="PATH",
        help=f"also save the route as a GPX file (default {GPX_PATH.name})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    lat, lon = postcode_to_latlon(POSTCODE)
    print(f"{POSTCODE} -> lat {lat:.5f}, lon {lon:.5f}")

    graph = get_network(lat, lon, ACTIVITY, RADIUS_M)
    start_node = ox.distance.nearest_nodes(graph, X=lon, Y=lat)

    choice = choose_route(
        graph,
        start_node,
        TARGET_M,
        min_depth_m=MIN_DEPTH_M,
        n_candidates=N_CANDIDATES,
        seed=SEED,
        radius_m=RADIUS_M,
    )
    print(format_choice(choice))
    polished = choice.chosen.polished

    quirks = polished.quirks_before
    spurs = sum(q.kind == "spur" for q in quirks)
    print(f"quirks before polishing: {spurs} spurs, {len(quirks) - spurs} zigzags")
    for quirk in quirks:
        print(f"  {describe_quirk(quirk)}")

    print(
        f"polish: {polished.hexes_before} -> {polished.hexes_after} hexagons, "
        f"{polished.length_before_m / 1000:.2f} -> {polished.length_after_m / 1000:.2f} km, "
        f"wasted {polished.wasted_before_m / 1000:.2f} -> {polished.wasted_after_m / 1000:.2f} km "
        f"({polished.rounds} improving rounds)"
    )

    # Other polishing steps can create new quirks, so this is not just "before minus removed".
    print(
        f"quirks: {len(polished.quirks_before)} before polishing, "
        f"{len(polished.quirks_kept)} left in the polished route"
    )
    for kept in polished.quirks_kept:
        hexes = f" (hexagons {', '.join(kept.hexes)})" if kept.hexes else ""
        print(f"  kept {describe_quirk(kept.quirk)} - {kept.reason}{hexes}")

    route_latlons = route_to_latlons(graph, polished.route)
    safe = hexes_for_track(route_latlons, MIN_DEPTH_M)
    risky = risky_hexes(route_latlons, MIN_DEPTH_M)
    path = draw_map((lat, lon), route_latlons, safe, OUT_PATH, risky_hex_ids=risky)
    print(f"{len(safe)} safe hexagons (at least {MIN_DEPTH_M} m inside), {polished.length_after_m / 1000:.2f} km")
    print(f"{len(risky)} risky hexagons (route only clips them, not counted)")
    print(f"saved {path}")
    if args.gpx is not None:
        name = f"Land Grab {POSTCODE} {TARGET_M / 1000:g} km {ACTIVITY}"
        gpx_path = save_gpx(route_latlons, args.gpx, name)
        print(f"saved {gpx_path} ({len(route_latlons)} points)")


if __name__ == "__main__":
    main()
