"""Command line: plan a hexagon-claiming loop from a postcode and save it as a map and/or GPX.

    python -m land_grab --postcode "SW1A 1AA" --km 10 --activity run --html map.html --gpx route.gpx
"""

import argparse
import math
import sys

import requests

from land_grab.hexgrid import RESOLUTION, parse_cells
from land_grab.pipeline import (  # noqa: F401  (network_radius_m etc. are re-exported)
    MAX_RADIUS_M,
    N_CANDIDATES,
    network_radius_m,
    plan_loop,
    save_loop_gpx,
    save_map,
    uncapped_network_radius_m,
)
from land_grab.planner import format_choice
from land_grab.router import LoopNotFoundError

ACTIVITIES = ("run", "walk", "ride")
MIN_KM = 1.0


def _km(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not (math.isfinite(value) and value >= MIN_KM):
        raise argparse.ArgumentTypeError(f"must be at least {MIN_KM:g} km, got {text}")
    return value


def _positive(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"must be positive, got {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m land_grab",
        description="Plan a loop from a UK postcode that claims as many hexagons as possible.",
    )
    parser.add_argument("--postcode", required=True, help='start postcode, e.g. "SW1A 1AA"')
    parser.add_argument("--km", required=True, type=_km, help=f"route length in km (at least {MIN_KM:g})")
    parser.add_argument("--activity", choices=ACTIVITIES, default="run", help="default: run")
    parser.add_argument(
        "--hex-size",
        type=_positive,
        default=None,
        help=f"ignored: hexagons are now the H3 grid at resolution {RESOLUTION}, whose size is fixed",
    )
    parser.add_argument(
        "--claimed",
        default="",
        metavar="IDS",
        help='H3 cell IDs you already own, comma-separated, e.g. "88194ad147fffff,88194ad141fffff"; '
        "they are not counted as new and the planner prefers other cells",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    parser.add_argument("--html", metavar="PATH", help="save an HTML map here")
    parser.add_argument("--gpx", metavar="PATH", help="save the route as a GPX file here")
    return parser


def run(args: argparse.Namespace) -> None:
    claimed = parse_cells(args.claimed)  # checked first, so a typo fails at once
    if args.hex_size is not None:
        print(f"note: --hex-size is ignored; the H3 grid (resolution {RESOLUTION}) has a fixed size", file=sys.stderr)
    loop = plan_loop(args.postcode, args.km, args.activity, claimed=claimed, seed=args.seed, on_status=print)
    print(format_choice(loop.choice))
    print(f"route length: {loop.length_m / 1000:.2f} km")
    if claimed:
        print(f"claimed cells (already yours, not counted): {len(claimed)}")
    print(f"new hexagons covered: {len(loop.safe_hexes)}")
    for problem in loop.problems:
        print(f"warning: {problem}", file=sys.stderr)

    saved = []
    if args.html:
        saved.append(save_map(loop, args.html))
    if args.gpx:
        saved.append(save_loop_gpx(loop, args.gpx))
    for path in saved:
        print(f"saved: {path}")
    if not saved:
        print("nothing saved (use --html and/or --gpx to save the route)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except (ValueError, LoopNotFoundError, requests.RequestException) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0
