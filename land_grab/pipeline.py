"""The whole pipeline in one call: postcode + distance -> planned loop, map and GPX.

Shared by the command line (``land_grab.cli``) and the Streamlit app (``app.py``, through
``land_grab.web``).
"""

import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import osmnx as ox

from land_grab.checks import Problem, validate_route
from land_grab.coverage import DEFAULT_MIN_DEPTH_M, hexes_for_track, risky_hexes
from land_grab.export import save_gpx
from land_grab.browser_streets import PHONE_FAILED_MESSAGE
from land_grab.geo import postcode_to_latlon
from land_grab.hexgrid import parse_cells
from land_grab.mapview import draw_map, route_to_latlons
from land_grab.network import BUSY_MESSAGE, FIRST_TIME_MESSAGE, get_network
from land_grab.planner import RouteChoice, choose_route, choose_route_fast
from land_grab.polish import PolishedRoute

N_CANDIDATES = 100
RADIUS_MARGIN = 0.10  # extra network beyond the furthest a loop can reach
MAX_RADIUS_M = 15_000.0  # cap, so a long bike ride does not trigger a huge download

# Progress stages reported through ``on_stage`` (the planner adds its own in between).
STAGE_POSTCODE = "Finding your postcode"
STAGE_STREETS = "Loading the streets"
STAGE_BUSY = BUSY_MESSAGE  # shown while a busy map server is retried
STAGE_FIRST_TIME = FIRST_TIME_MESSAGE  # shown before a slow download of streets no ready-made file covers
STAGE_PHONE_FAILED = PHONE_FAILED_MESSAGE  # shown when the phone's download did not work and our server takes over


class InvalidPostcodeError(ValueError):
    """The postcode is empty or was not found."""


def uncapped_network_radius_m(km: float) -> float:
    """Radius needed to hold every loop of ``km`` km, before the cap is applied.

    A loop can never get further from its start than half its length, plus a
    10% margin. One-way routes (not supported yet) can get the full distance
    away, so they will need ``km * 1000`` here instead.
    """
    return km / 2 * 1000 * (1 + RADIUS_MARGIN)


def network_radius_m(km: float) -> float:
    """Download radius in metres for a loop of ``km`` km: half its length plus 10%, at most 15 km."""
    if not km > 0:
        raise ValueError(f"km must be positive, got {km}")
    return min(uncapped_network_radius_m(km), MAX_RADIUS_M)


@dataclass(frozen=True)
class PlannedLoop:
    postcode: str
    km: float
    activity: str
    start: tuple[float, float]  # (lat, lon)
    radius_m: float
    radius_capped: bool
    choice: RouteChoice
    route_latlons: list
    safe_hexes: set  # new cells the route claims (cells you already own are not in here)
    risky_hexes: set
    claimed: frozenset = frozenset()  # cells you said you already own

    @property
    def polished(self) -> PolishedRoute:
        return self.choice.chosen.polished

    @property
    def length_m(self) -> float:
        return self.polished.length_after_m

    @property
    def n_points(self) -> int:
        return len(self.route_latlons)

    @property
    def gpx_name(self) -> str:
        return f"Land Grab {self.postcode.strip()} {self.km:g} km {self.activity}"

    @property
    def problems(self) -> list[Problem]:
        """What ``validate_route`` finds wrong with the route (empty if it looks fine)."""
        return validate_route(self.route_latlons, self.km, self.activity)


def plan_loop(
    postcode: str,
    km: float,
    activity: str = "run",
    *,
    claimed: str | Iterable[str] | None = None,
    seed: int = 0,
    n_candidates: int | None = None,
    on_status: Callable[[str], None] | None = None,
    planner_only: bool = False,
    on_stage: Callable[[str], None] | None = None,
    ask_phone: bool = False,
    phone_reply: dict | None = None,
) -> PlannedLoop:
    """Plan a loop of ``km`` km from a postcode.

    By default the planner and the random search both run and the better one
    is kept (``choose_route``). With ``planner_only`` only the planner runs, and
    the random search is a fallback for loops of 12 km or less
    (``choose_route_fast``); the web app uses this because it is much faster.

    ``on_status`` gets a short technical message at each stage (the CLI prints
    them); ``on_stage`` gets a plain progress stage such as "Loading the
    streets" (the web app shows them). ``claimed`` is H3 cell IDs you already
    own, as comma-separated text or a list: they are drawn in their own colour,
    not counted as new, and avoided where possible. Raises InvalidPostcodeError
    for an empty or unknown postcode, InvalidCellError for bad cell IDs, ValueError for other bad arguments,
    LoopNotFoundError if no route is found, and requests.RequestException if
    the postcode service cannot be reached.

    ``ask_phone`` (the web app) makes this raise ``NeedPhoneStreets`` when no saved or ready-made file
    covers the place, so the page can have the visitor's phone download the streets; the planning is then
    run again with the phone's answer as ``phone_reply`` (see ``network.get_network``). The command line
    uses neither and downloads on this computer, as before.
    """
    say = on_status or (lambda message: None)
    stage = on_stage or (lambda name: None)
    radius_m = network_radius_m(km)  # validates km before anything is downloaded
    claimed_cells = parse_cells(claimed)  # so bad cell IDs are reported before anything is downloaded

    stage(STAGE_POSTCODE)
    try:
        lat, lon = postcode_to_latlon(postcode)
    except ValueError as error:
        raise InvalidPostcodeError(str(error)) from error
    say(f"start: {lat:.5f}, {lon:.5f}")

    capped = uncapped_network_radius_m(km) > MAX_RADIUS_M
    note = f" (capped at {MAX_RADIUS_M / 1000:g} km; the loop may not reach everywhere it could)" if capped else ""
    say(f"network radius: {radius_m / 1000:.2f} km{note}")

    stage(STAGE_STREETS)

    def busy() -> None:
        say(STAGE_BUSY)
        stage(STAGE_BUSY)

    def first_time() -> None:
        say(STAGE_FIRST_TIME)
        stage(STAGE_FIRST_TIME)

    def phone_failed(reason: str) -> None:
        say(f"{STAGE_PHONE_FAILED} ({reason})")
        stage(STAGE_PHONE_FAILED)

    extra = {}  # only passed when used, so the command line calls get_network exactly as before
    if ask_phone:
        extra["ask_phone"] = True
    if phone_reply is not None:
        extra.update(phone_reply=phone_reply, on_phone_failed=phone_failed)
    graph = get_network(lat, lon, activity, radius_m, on_busy=busy, on_notice=say, on_slow=first_time, **extra)
    start_node = ox.distance.nearest_nodes(graph, X=lon, Y=lat)

    options = dict(
        claimed=claimed_cells,
        min_depth_m=DEFAULT_MIN_DEPTH_M,
        n_candidates=N_CANDIDATES if n_candidates is None else n_candidates,
        seed=seed,
        radius_m=radius_m,
    )
    if planner_only:
        choice = choose_route_fast(graph, start_node, km * 1000, on_stage=on_stage, **options)
    else:
        choice = choose_route(graph, start_node, km * 1000, **options)
    route_latlons = route_to_latlons(graph, choice.chosen.polished.route)
    return PlannedLoop(
        postcode=postcode,
        km=km,
        activity=activity,
        start=(lat, lon),
        radius_m=radius_m,
        radius_capped=capped,
        choice=choice,
        route_latlons=route_latlons,
        safe_hexes=hexes_for_track(route_latlons, DEFAULT_MIN_DEPTH_M) - claimed_cells,
        risky_hexes=risky_hexes(route_latlons, DEFAULT_MIN_DEPTH_M) - claimed_cells,
        claimed=claimed_cells,
    )


def save_map(loop: PlannedLoop, out_path: str | Path) -> Path:
    return draw_map(
        loop.start,
        loop.route_latlons,
        loop.safe_hexes,
        out_path,
        risky_hex_ids=loop.risky_hexes,
        claimed_hex_ids=loop.claimed,
    )


def save_loop_gpx(loop: PlannedLoop, out_path: str | Path) -> Path:
    return save_gpx(loop.route_latlons, out_path, loop.gpx_name)


def map_html(loop: PlannedLoop) -> str:
    """The map as an HTML string (for embedding)."""
    with tempfile.TemporaryDirectory() as folder:
        return save_map(loop, Path(folder) / "map.html").read_text(encoding="utf-8")


def gpx_text(loop: PlannedLoop) -> str:
    """The route as GPX XML text (for a download button)."""
    with tempfile.TemporaryDirectory() as folder:
        return save_loop_gpx(loop, Path(folder) / "route.gpx").read_text(encoding="utf-8")
