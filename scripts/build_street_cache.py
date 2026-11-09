"""Make the ready-made street files in ./prebuilt, so the web app does not have to download streets live.

Needs internet access (postcodes.io and the OpenStreetMap Overpass servers) and takes a while: each area is
a big download, and the public servers are often busy, so a busy answer is retried. Run it from the project
root on your own computer:

    .venv/bin/python scripts/build_street_cache.py

Useful options:

    --only london-se17          build just the areas whose name contains this text
    --activity run              build just "run" (or "ride")
    --force                     build again even if the file already exists

For each area and activity it saves one compressed file, named like ``london-se17-10km_run.pkl.gz``, and
prints how big the file is. The files are NOT kept in git (see DEPLOY.md).
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from land_grab.geo import postcode_to_latlon  # noqa: E402
from land_grab.network import (  # noqa: E402
    NETWORK_TYPES,
    PREBUILT_DIR,
    download_graph,
    prebuilt_name,
    remove_excluded_roads,
    save_prebuilt,
)

# (short name for the file, postcode in the middle, radius in metres)
AREAS = [
    ("london-se17", "SE17 3BA", 10_000),
    ("london-sw1a", "SW1A 1AA", 5_000),
]
ACTIVITIES = ("run", "ride")  # "run" files also serve "walk": both use the same streets

# Longer waits than the web app uses: nobody is watching, and a big download is worth waiting for.
WAITS_S = (15, 45, 90, 180)


def size_text(num_bytes: int) -> str:
    return f"{num_bytes / 1_000_000:.1f} MB"


def build(area: str, postcode: str, radius_m: int, activity: str, folder: Path, force: bool) -> int:
    """Download, clean and save one file. Returns its size in bytes (0 if it was skipped)."""
    path = folder / prebuilt_name(area, radius_m, activity)
    if path.exists() and not force:
        print(f"  {path.name}: already there ({size_text(path.stat().st_size)}), skipped (use --force to rebuild)")
        return 0
    lat, lon = postcode_to_latlon(postcode)
    print(f"  {path.name}: downloading {radius_m / 1000:g} km around {postcode} ({lat:.5f}, {lon:.5f}) ...", flush=True)
    began = time.perf_counter()
    graph = download_graph(
        lat, lon, NETWORK_TYPES[activity], radius_m, on_busy=lambda: print("    the map service is busy, waiting ..."), waits=WAITS_S
    )
    graph = remove_excluded_roads(graph)
    save_prebuilt(graph, path, lat, lon, radius_m, activity, area)
    size = path.stat().st_size
    print(
        f"  {path.name}: {size_text(size)}  ({graph.number_of_nodes():,} street points, "
        f"{graph.number_of_edges():,} street pieces, {time.perf_counter() - began:.0f} s)"
    )
    return size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download street networks for named areas into ./prebuilt.")
    parser.add_argument("--only", help="only areas whose name contains this text")
    parser.add_argument("--activity", choices=ACTIVITIES, help="only this activity")
    parser.add_argument("--force", action="store_true", help="build again even if the file exists")
    parser.add_argument("--folder", type=Path, default=PREBUILT_DIR, help="where to save (default: ./prebuilt)")
    args = parser.parse_args(argv)

    jobs = [
        (name, postcode, radius, activity)
        for name, postcode, radius in AREAS
        if not args.only or args.only in name
        for activity in ACTIVITIES
        if not args.activity or args.activity == activity
    ]
    if not jobs:
        print("Nothing matches those options.")
        return 1

    print(f"Building {len(jobs)} file(s) into {args.folder}")
    failed = 0
    for name, postcode, radius, activity in jobs:
        try:
            build(name, postcode, radius, activity, args.folder, args.force)
        except Exception as error:  # keep going: one busy area should not lose the others
            failed += 1
            print(f"  {prebuilt_name(name, radius, activity)}: FAILED - {type(error).__name__}: {error}")

    print("\nFiles in", args.folder)
    total = 0
    for path in sorted(args.folder.glob("*.pkl.gz")):
        total += path.stat().st_size
        print(f"  {size_text(path.stat().st_size):>10}  {path.name}")
    print(f"  {size_text(total):>10}  total")
    if failed:
        print(f"\n{failed} file(s) failed. Run the same command again: finished files are skipped.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
