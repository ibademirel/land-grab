"""What the web app needs besides the planner: a route cache, progress stages and friendly errors.

Kept free of Streamlit so it can be tested without it. ``app.py`` is only the screen.
"""

import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import requests
from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError

from land_grab import pipeline
from land_grab.browser_streets import NeedPhoneStreets  # noqa: F401  (app.py imports it from here)
from land_grab.hexgrid import RESOLUTION, InvalidCellError, hexes_within_radius, is_valid_hex, parse_cells
from land_grab.network import MapServiceBusyError
from land_grab.pipeline import (  # noqa: F401  (app.py imports the STAGE_ names from here)
    STAGE_BUSY,
    STAGE_FIRST_TIME,
    STAGE_PHONE_FAILED,
    InvalidPostcodeError,
    PlannedLoop,
)
from land_grab.planner import SKIPPED, format_choice
from land_grab.router import LoopNotFoundError

ACTIVITIES = ("run", "walk", "ride")

# The longest route the WEB APP will plan, in km. The free Streamlit Community Cloud server has about
# 1 GB of memory, and the street map of a big area does not fit: measured on a Mac, loading the streets
# for a 5.5 km radius (a 10 km run, 81,000 street points) takes about 360 MB, and a 15 km radius (a
# 27 km ride, 229,000 points) about 870 MB before any planning. A 15 km ride needs a radius of about
# 8 km, which is about the same size as the 10 km run, so these three limits keep to the safe side.
# The command line has no such limit. Raise these only on a bigger server.
MAX_KM = {"run": 10.0, "walk": 10.0, "ride": 15.0}
CACHE_SIZE = 64  # finished routes kept; the oldest unused one goes first
STAGE_MAP = "Drawing your map"
BUSY_FAILED_MESSAGE = "The map service is busy right now. Please wait a minute and press Plan again."

RouteKey = tuple[str, float, str, frozenset]  # (postcode without spaces, uppercase; km; activity; claimed cells)


def cache_key(postcode: str, km: float, activity: str, claimed: str | None = None) -> RouteKey:
    """The same route asked for in a different way ("sw1a1aa", "SW1A 1AA ") gets the same key.

    The claimed cells are part of the key (in any order or spelling), because they change the route.
    Raises InvalidCellError if they are not cell IDs.
    """
    return "".join(postcode.split()).upper(), float(km), activity.strip().lower(), parse_cells(claimed)


class TooFarError(ValueError):
    """The web app does not plan routes this long (the message is written for people)."""


def check_distance(km: float, activity: str) -> None:
    """Raise TooFarError if ``km`` is more than the web app plans for this activity."""
    limit = MAX_KM.get(activity.strip().lower())
    if limit is not None and float(km) > limit:
        raise TooFarError(
            f"That's further than this web version can plan: up to {limit:g} km for a "
            f"{activity.strip().lower()}. Try {limit:g} km or less. "
            "(The command line version has no limit.)"
        )


# --- the whole data block copied from the challenge page ---------------------------------

_CELLS_LIST = re.compile(r"""["']?\bcells\b["']?\s*[:=]\s*\[([^\]]*)""", re.IGNORECASE)  # everything up to the closing ]
_CELL_CODE = re.compile(r"(?<![0-9a-z])8[0-9a-f]{14}(?![0-9a-z])", re.IGNORECASE)  # 15 hex characters, the first is 8


def cells_from_block(text: str | None) -> list[str]:
    """Pick the claimed cell codes out of a whole pasted data block, ignoring everything else.

    Only the list under the "cells" key is read (the text from its opening bracket to the
    closing one), so teams, activities, regions and any other codes are ignored. If "cells"
    appears more than once, the one that is least nested is used (the block's own, not one
    inside a team). Repeats are removed and the order is kept. Returns [] when nothing is found.
    """
    text = text or ""
    matches = list(_CELLS_LIST.finditer(text))
    if not matches:
        return []

    def nesting(match: re.Match) -> int:
        before = text[: match.start()]
        return before.count("{") + before.count("[") - before.count("}") - before.count("]")

    best = min(matches, key=nesting)  # min keeps the first of equals
    codes = dict.fromkeys(code.lower() for code in _CELL_CODE.findall(best.group(1)))
    return [code for code in codes if is_valid_hex(code)]


def join_claimed(typed: str | None, block: str | None) -> str:
    """The one claimed-cells text the planner gets: the single-line box plus the cells found in the pasted block."""
    return ", ".join(part for part in [(typed or "").strip(), ", ".join(cells_from_block(block))] if part)


def cells_outside_area(loop: PlannedLoop) -> frozenset:
    """Claimed cells that lie beyond the streets we loaded around the start, so they cannot touch this route."""
    lat, lon = loop.start
    return frozenset(loop.claimed - hexes_within_radius(lat, lon, loop.radius_m))


@dataclass(frozen=True)
class RouteResult:
    """A finished route with everything the screen shows."""

    loop: PlannedLoop
    map_html: str
    gpx: str
    seconds: float  # how long it took to make (not to fetch from the cache)

    @property
    def hexagons(self) -> int:
        return len(self.loop.safe_hexes)

    @property
    def distance_km(self) -> float:
        return self.loop.length_m / 1000

    @property
    def points(self) -> int:
        return self.loop.n_points


class RouteCache:
    """A small thread-safe least-recently-used cache of finished routes."""

    def __init__(self, max_size: int = CACHE_SIZE):
        if max_size < 1:
            raise ValueError(f"max_size must be at least 1, got {max_size}")
        self.max_size = max_size
        self._items: OrderedDict[RouteKey, RouteResult] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: RouteKey) -> RouteResult | None:
        with self._lock:
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key]

    def put(self, key: RouteKey, result: RouteResult) -> None:
        with self._lock:
            self._items[key] = result
            self._items.move_to_end(key)
            while len(self._items) > self.max_size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: RouteKey) -> bool:
        return key in self._items


ROUTES = RouteCache()  # shared by everyone using the running app


def get_route(
    postcode: str,
    km: float,
    activity: str,
    claimed: str | None = None,
    *,
    on_stage: Callable[[str], None] | None = None,
    cache: RouteCache | None = None,
    ask_phone: bool = False,
    phone_reply: dict | None = None,
) -> tuple[RouteResult, bool]:
    """Return (route, came_from_cache) for a postcode, distance, activity and claimed cells.

    ``claimed`` is pasted text of H3 cell IDs you already own (see ``hexgrid.parse_cells``).

    A route asked for before comes straight from the cache, with no lookup and
    no planning, and reports no stages. Otherwise the planner runs on its own
    (see ``pipeline.plan_loop``, ``planner_only``; the random search is only a
    fallback for routes of 12 km or less), ``on_stage`` hears about each step,
    and the finished route is cached. Failures are never cached.

    ``ask_phone`` lets the visitor's phone download streets that no ready-made file covers: then this raises
    ``NeedPhoneStreets`` (not an error; the page hands its request to the phone) and is called again with
    the phone's answer as ``phone_reply``. See ``network.get_network``.

    Raises TooFarError for a distance over ``MAX_KM`` (web only; the command line is unlimited),
    and otherwise what ``plan_loop`` raises; ``friendly_error`` turns that into words.
    """
    cache = ROUTES if cache is None else cache
    key = cache_key(postcode, km, activity, claimed)
    check_distance(km, activity)
    found = cache.get(key)
    if found is not None:
        return found, True

    began = time.perf_counter()
    loop = pipeline.plan_loop(
        postcode,
        float(km),
        activity.strip().lower(),
        claimed=key[3],
        planner_only=True,
        on_stage=on_stage,
        ask_phone=ask_phone,
        phone_reply=phone_reply,
    )
    if on_stage:
        on_stage(STAGE_MAP)
    result = RouteResult(loop, pipeline.map_html(loop), pipeline.gpx_text(loop), time.perf_counter() - began)
    cache.put(key, result)
    return result, False


def friendly_error(error: Exception, postcode: str = "") -> str:
    """A plain, kind sentence or two for anything that can go wrong."""
    if isinstance(error, (InvalidCellError, TooFarError)):
        return str(error)
    if isinstance(error, InvalidPostcodeError):
        if not postcode.strip():
            return "Please type a postcode to start from, for example SW1A 1AA."
        return f"Sorry, we couldn't find the postcode \"{postcode.strip()}\". Check it and try again, for example SW1A 1AA."
    if isinstance(error, LoopNotFoundError):
        return "We couldn't find a route of that length from there. Try a different distance or another postcode."
    # OSMnx's "the server answered with an error" is a kind of ValueError, so these two must come
    # before the ValueError case below, which is only meant for a bad distance.
    if isinstance(error, (MapServiceBusyError, ResponseStatusCodeError)):
        return BUSY_FAILED_MESSAGE
    if isinstance(error, InsufficientResponseError):
        return "The map service didn't send back any streets for that place. Please try again in a minute."
    if isinstance(error, requests.RequestException):
        return "We couldn't reach the postcode or map service. Check your internet connection and try again."
    if isinstance(error, ValueError):
        return "That distance doesn't look right. Please pick a number of kilometres, 1 or more."
    return "Something went wrong while planning your route. Please try again."


def details_text(result: RouteResult, came_from_cache: bool = False) -> str:
    """The technical detail for the collapsed "Details" section."""
    loop = result.loop
    lat, lon = loop.start
    lines = [
        format_choice(loop.choice),
        "",
        f"start: {lat:.5f}, {lon:.5f}",
        f"streets loaded within {loop.radius_m / 1000:.2f} km"
        + (" (capped, so a loop may not reach everywhere it could)" if loop.radius_capped else ""),
        f"grid: H3 resolution {RESOLUTION}",
        f"new safe hexagons: {len(loop.safe_hexes)}, risky (clipped only, not counted): {len(loop.risky_hexes)}",
    ]
    if loop.claimed:
        lines.append(f"cells already claimed (not counted as new): {len(loop.claimed)}")
    if loop.choice.random.error != SKIPPED:
        lines.append("the planner could not make this route, so the random search was used")
    lines.append(
        "shown instantly from an earlier search"
        if came_from_cache
        else f"made in {result.seconds:.1f} s"
    )
    return "\n".join(lines)
