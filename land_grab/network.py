"""Street networks: ready-made files in ``prebuilt/``, else a download from OpenStreetMap (via OSMnx).

``get_network`` looks for the streets in this order: the local pickle cache, then a ready-made file in
``prebuilt/`` that covers the request (made by ``scripts/build_street_cache.py``), then the same kind of
file downloaded from the project's GitHub Release (``RELEASE_FILES``), and only then the public
Overpass servers.
"""

import gzip
import hashlib
import logging
import os
import pickle
import tempfile
import threading
import time
from dataclasses import dataclass
from collections.abc import Callable
from datetime import date
from pathlib import Path

import networkx as nx
import osmnx as ox
import requests

log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent
PREBUILT_DIR = PROJECT_DIR / "prebuilt"  # ready-made street files; not kept in git (see DEPLOY.md)


def usable_cache_dir(preferred: Path = PROJECT_DIR / "cache") -> Path:
    """The folder for cached street data, created if it is missing.

    It sits next to the project (not in whatever folder the program was started
    from), so it is the same place on a laptop and on a web server. The folder
    is not kept in git, so a fresh copy of the project (such as on Streamlit
    Community Cloud) starts without one: it is made here. If the project folder
    cannot be written to, a temporary folder is used instead.
    """
    for folder in (preferred, Path(tempfile.gettempdir()) / "land_grab_cache"):
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(folder, os.W_OK):
            return folder
    return preferred  # nothing is writable: downloads still work, they just are not saved


CACHE_DIR = usable_cache_dir()
ox.settings.cache_folder = str(CACHE_DIR)  # OSMnx keeps its own downloaded answers here
ox.settings.use_cache = True

# --- a busy map server -------------------------------------------------------------------
# OpenStreetMap's street data comes from free public "Overpass" servers, which are often busy.

BUSY_MESSAGE = "Map service is busy, trying again..."
FIRST_TIME_MESSAGE = (
    "First time planning in this area. Loading the street map can take 1 to 3 minutes, "
    "so please keep this page open. Central and south London load instantly."
)
BUSY_STATUS_CODES = frozenset({429, 502, 503, 504})  # too many requests, bad gateway, unavailable, timed out
RETRY_WAITS_S = (5, 15, 30)  # the first server is retried 3 times, waiting longer each time

# Public servers, tried in this order. (overpass.kumi.systems is now overpass.private.coffee, so it is not listed twice.)
OVERPASS_SERVERS = (
    "https://overpass-api.de/api",
    "https://overpass.private.coffee/api",
    "https://maps.mail.ru/osm/tools/overpass/api",
)

_download_lock = threading.Lock()  # one street download at a time: kinder to the servers and to the memory


class MapServiceBusyError(requests.RequestException):
    """A map server answered 429, 502, 503 or 504 (the message is for the details box)."""

    def __init__(self, status_code: int, url: str):
        self.status_code = status_code
        super().__init__(f"{url.split('/')[2] if '//' in url else url} responded {status_code}: the map service is busy")


def _raise_if_busy(response: requests.Response, *args, **kwargs) -> requests.Response:
    """A requests "response hook": stop at a busy answer, before OSMnx waits 55 s or tries to read it."""
    if response.status_code in BUSY_STATUS_CODES:
        raise MapServiceBusyError(response.status_code, response.url or "")
    return response


# --- which streets we ask for ------------------------------------------------------------------
# One definition, used by the live download AND by the script that makes the ready-made files, so the
# two always agree. These are OSMnx's own "walk" and "bike" rules (public ways a person on foot, or on a
# bike, may use) plus one more rule: no motorways or trunk roads, which the challenge forbids. Asking
# Overpass not to send them saves time and memory; ``remove_excluded_roads`` below is the safety net.

_EXCLUDED_REGEX = "|".join(sorted(["motorway", "motorway_link", "trunk", "trunk_link"]))
_NO_MOTORWAYS = f'["highway"!~"{_EXCLUDED_REGEX}"]'
_STREET_FILTERS = {
    "walk": (
        '["highway"]["area"!~"yes"]["access"!~"private"]'
        '["highway"!~"abandoned|bus_guideway|construction|cycleway|motor|no|planned|'
        'platform|proposed|raceway|razed|rest_area|services"]'
        '["foot"!~"no"]["service"!~"private"]'
        '["sidewalk"!~"separate"]["sidewalk:both"!~"separate"]'
        '["sidewalk:left"!~"separate"]["sidewalk:right"!~"separate"]'
    ),
    "bike": (
        '["highway"]["area"!~"yes"]["access"!~"private"]'
        '["highway"!~"abandoned|bus_guideway|construction|corridor|elevator|'
        'escalator|footway|motor|no|planned|platform|proposed|raceway|razed|'
        'rest_area|services|steps"]'
        '["bicycle"!~"no"]["service"!~"private"]'
    ),
}


def street_filter(network_type: str) -> str:
    """The Overpass filter for the street types the planner uses: walkable or cyclable, no motorways."""
    try:
        return _STREET_FILTERS[network_type] + _NO_MOTORWAYS
    except KeyError:
        raise ValueError(f"Unknown network type {network_type!r}; expected one of {sorted(_STREET_FILTERS)}") from None


def filter_fingerprint(network_type: str) -> str:
    """A short code for the filter. A ready-made file made with another filter has another code and is not used."""
    return hashlib.sha1(street_filter(network_type).encode()).hexdigest()[:12]


def download_graph(
    lat: float,
    lon: float,
    network_type: str,
    radius_m: float,
    on_busy: Callable[[], None] | None = None,
    servers: tuple[str, ...] = OVERPASS_SERVERS,
    waits: tuple[float, ...] = RETRY_WAITS_S,
) -> nx.MultiDiGraph:
    """Download the streets, coping with a busy server.

    The first server is tried, then retried once after each wait in ``waits`` if it is busy. If it is
    still busy, each of the other servers gets one try. A server that cannot be reached at all is skipped
    straight away. ``on_busy`` is called just before every wait or switch. If every server fails the last
    busy error is raised (or else the last connection error).
    """
    notify = on_busy or (lambda: None)
    busy_error: Exception | None = None
    other_error: Exception | None = None

    with _download_lock:
        old_server, old_kwargs = ox.settings.overpass_url, ox.settings.requests_kwargs
        kwargs = dict(old_kwargs)
        kwargs["hooks"] = {"response": [*dict(old_kwargs.get("hooks", {})).get("response", []), _raise_if_busy]}
        ox.settings.requests_kwargs = kwargs
        try:
            for index, server in enumerate(servers):
                ox.settings.overpass_url = server
                tries = len(waits) + 1 if index == 0 else 1
                for attempt in range(tries):
                    try:
                        # network_type is still passed: OSMnx uses it to make "walk" streets two-way.
                        return ox.graph_from_point(
                            (lat, lon), dist=radius_m, network_type=network_type, custom_filter=street_filter(network_type)
                        )
                    except MapServiceBusyError as error:
                        busy_error = error
                    except (requests.ConnectionError, requests.Timeout) as error:
                        other_error = error
                        break  # this server is not answering at all: go to the next one
                    if attempt < tries - 1:
                        notify()
                        time.sleep(waits[attempt])
                if index < len(servers) - 1:
                    notify()
        finally:
            ox.settings.overpass_url, ox.settings.requests_kwargs = old_server, old_kwargs

    raise busy_error or other_error or MapServiceBusyError(503, "no map server")


# Activity -> OSMnx network_type
NETWORK_TYPES = {
    "run": "walk",
    "walk": "walk",
    "ride": "bike",
}

# Roads a runner, walker or cyclist may not use (and the challenge forbids).
EXCLUDED_HIGHWAYS = frozenset({"motorway", "motorway_link", "trunk", "trunk_link"})

# Coordinates are rounded to 4 decimals (about 11 m) for the cache key.
COORD_DECIMALS = 4


def _network_type(activity: str) -> str:
    try:
        return NETWORK_TYPES[activity.strip().lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"Unknown activity {activity!r}; expected one of {sorted(NETWORK_TYPES)}"
        ) from None


def cache_key(lat: float, lon: float, activity: str, radius_m: float) -> str:
    """Return a stable, filesystem-safe cache key, e.g. ``run_51.5010_-0.1416_3000``."""
    activity = activity.strip().lower()
    _network_type(activity)  # validates
    return f"{activity}_{lat:.{COORD_DECIMALS}f}_{lon:.{COORD_DECIMALS}f}_{round(radius_m)}"


def cache_path(
    lat: float,
    lon: float,
    activity: str,
    radius_m: float,
    cache_dir: Path = CACHE_DIR,
) -> Path:
    return Path(cache_dir) / f"{cache_key(lat, lon, activity, radius_m)}.pkl"


def _is_excluded(highway) -> bool:
    """OSMnx stores the highway tag as a string, or a list when edges were merged."""
    values = highway if isinstance(highway, (list, tuple, set)) else [highway]
    return any(value in EXCLUDED_HIGHWAYS for value in values)


def remove_excluded_roads(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Return a copy of the graph without motorway/trunk edges or the nodes they leave stranded."""
    cleaned = graph.copy()
    to_remove = [
        (u, v, k)
        for u, v, k, data in cleaned.edges(keys=True, data=True)
        if _is_excluded(data.get("highway"))
    ]
    cleaned.remove_edges_from(to_remove)
    cleaned.remove_nodes_from(list(nx.isolates(cleaned)))
    return cleaned


# --- ready-made street files ---------------------------------------------------------------
# Each file in ``prebuilt/`` holds the streets of one named area for one activity, squeezed with gzip.
# Inside, a small "label" comes first (where, how big, which filter), then the streets, so the label can
# be read without unpacking the streets. Only the street data the planner uses is kept.

PREBUILT_SUFFIX = ".pkl.gz"
PREBUILT_FORMAT = 1  # bump this if the layout of the files changes; older files are then ignored
PREBUILT_DAMAGED_MESSAGE = (
    "One of the ready-made street files looked damaged, so the streets are being downloaded instead. "
    "This may take a little longer."
)


def prebuilt_name(area: str, radius_m: float, activity: str) -> str:
    """A clear file name, for example ``london-se17-10km_run.pkl.gz``."""
    activity = activity.strip().lower()
    _network_type(activity)  # validates
    return f"{area}-{radius_m / 1000:g}km_{activity}{PREBUILT_SUFFIX}"


def slim_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """A copy with only what the planner reads: node x/y; edge length, highway and (if any) shape; the CRS.

    Street names, speed limits, OSM ids and the like are dropped, which makes the file and the memory
    use a good deal smaller.
    """
    slim = nx.MultiDiGraph(crs=graph.graph.get("crs", ox.settings.default_crs))
    slim.add_nodes_from((node, {"x": float(data["x"]), "y": float(data["y"])}) for node, data in graph.nodes(data=True))
    for u, v, key, data in graph.edges(keys=True, data=True):
        kept = {"length": float(data["length"])}
        if "highway" in data:
            kept["highway"] = data["highway"]
        if data.get("geometry") is not None:
            kept["geometry"] = data["geometry"]
        slim.add_edge(u, v, key, **kept)
    return slim


def save_prebuilt(
    graph: nx.MultiDiGraph, path: Path, lat: float, lon: float, radius_m: float, activity: str, area: str = ""
) -> Path:
    """Write a compressed street file (label first, then the slimmed streets). Returns the path."""
    network_type = _network_type(activity)
    slim = slim_graph(graph)
    label = {
        "format": PREBUILT_FORMAT,
        "area": area,
        "activity": activity.strip().lower(),
        "network_type": network_type,
        "filter": filter_fingerprint(network_type),
        "lat": float(lat),
        "lon": float(lon),
        "radius_m": float(radius_m),
        "built": date.today().isoformat(),
        "nodes": slim.number_of_nodes(),
        "edges": slim.number_of_edges(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with gzip.open(tmp_path, "wb", compresslevel=6) as f:
        pickle.dump(label, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(slim, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)  # atomic, like the local cache
    return path


def _bbox(lat: float, lon: float, radius_m: float) -> tuple[float, float, float, float]:
    """(left, bottom, right, top) in degrees: the square OSMnx cuts out for a point and a radius."""
    return tuple(float(v) for v in ox.utils_geo.bbox_from_point((lat, lon), radius_m))


def _label_covers(label: dict, lat: float, lon: float, radius_m: float, network_type: str) -> bool:
    """True if the file's area contains the whole requested square and its filter and layout are current."""
    if label["format"] != PREBUILT_FORMAT or label["network_type"] != network_type:
        return False
    if label["filter"] != filter_fingerprint(network_type):
        return False  # made with other street types than the ones we ask for now
    left, bottom, right, top = _bbox(lat, lon, radius_m)
    have_left, have_bottom, have_right, have_top = _bbox(label["lat"], label["lon"], label["radius_m"])
    return have_left <= left and have_bottom <= bottom and right <= have_right and top <= have_top


def _trim_to_request(graph: nx.MultiDiGraph, lat: float, lon: float, radius_m: float) -> nx.MultiDiGraph:
    """Keep only the streets inside the requested square (and, if anything was cut, the main connected part).

    The graph is cut down in place rather than copied, so the memory of the big one is given back
    instead of briefly needing both. (It is a freshly loaded graph that nobody else holds.)
    """
    left, bottom, right, top = _bbox(lat, lon, radius_m)
    outside = [n for n, d in graph.nodes(data=True) if not (left <= d["x"] <= right and bottom <= d["y"] <= top)]
    if not outside:
        return graph
    graph.remove_nodes_from(outside)
    if graph.number_of_nodes():
        main = max(nx.weakly_connected_components(graph), key=len)
        if len(main) < graph.number_of_nodes():
            graph.remove_nodes_from([n for n in graph if n not in main])
    return graph


def find_prebuilt(
    lat: float,
    lon: float,
    activity: str,
    radius_m: float,
    prebuilt_dir: Path,
    on_notice: Callable[[str], None] | None = None,
) -> nx.MultiDiGraph | None:
    """The streets from a ready-made file that covers the request, or None if no file does.

    Only each file's small label is read to decide; the one chosen (the smallest area that covers the
    request, so the least to load) is then unpacked and trimmed to the requested square. A damaged or
    unreadable file is skipped with a friendly message (``on_notice`` and the log) and never stops the
    caller: it just falls back to a download.
    """
    network_type = _network_type(activity)
    folder = Path(prebuilt_dir)
    if not folder.is_dir():
        return None

    covering: list[tuple[float, Path]] = []
    for path in sorted(folder.glob(f"*{PREBUILT_SUFFIX}")):
        try:
            with gzip.open(path, "rb") as f:
                label = pickle.load(f)
            if _label_covers(label, lat, lon, radius_m, network_type):
                covering.append((label["radius_m"], path))
        except Exception as error:  # damaged, cut short, or not one of ours
            _say_damaged(path, error, on_notice)

    for _, path in sorted(covering):
        try:
            with gzip.open(path, "rb") as f:
                pickle.load(f)  # the label again, to get to the streets behind it
                graph = pickle.load(f)
            if not isinstance(graph, nx.MultiDiGraph) or "crs" not in graph.graph:
                raise ValueError("not a street network")
            return _trim_to_request(graph, lat, lon, radius_m)
        except Exception as error:
            _say_damaged(path, error, on_notice)
    return None


def _say_damaged(path: Path, error: Exception, on_notice: Callable[[str], None] | None) -> None:
    log.warning("Skipping prebuilt street file %s: %s: %s", path.name, type(error).__name__, error)
    if on_notice:
        on_notice(PREBUILT_DAMAGED_MESSAGE)


# --- ready-made files published on GitHub ----------------------------------------------------
# The files made by scripts/build_street_cache.py are published as a GitHub Release. When the request is
# inside one of these areas and the file is not on this computer yet, it is downloaded once into
# ``prebuilt/`` (a few seconds) instead of asking Overpass for all those streets.


@dataclass(frozen=True)
class ReleaseFile:
    """A published file and, roughly, the area it covers (the exact area is read from the file itself)."""

    url: str
    activity: str  # "run" or "ride"
    lat: float  # about the middle of the area (the postcode's position, to 4 decimals)
    lon: float
    radius_m: float

    @property
    def name(self) -> str:
        return self.url.rsplit("/", 1)[1]

    @property
    def network_type(self) -> str:
        return _network_type(self.activity)


_RELEASE_URL = "https://github.com/ibademirel/land-grab/releases/download/streets-v1/"
_SE17 = (51.4911, -0.0986)  # SE17 3BA
_SW1A = (51.5010, -0.1416)  # SW1A 1AA

RELEASE_FILES: tuple[ReleaseFile, ...] = (
    ReleaseFile(_RELEASE_URL + "london-se17-10km_run.pkl.gz", "run", *_SE17, 10_000),
    ReleaseFile(_RELEASE_URL + "london-se17-10km_ride.pkl.gz", "ride", *_SE17, 10_000),
    ReleaseFile(_RELEASE_URL + "london-sw1a-5km_ride.pkl.gz", "ride", *_SW1A, 5_000),
)

DOWNLOAD_TIMEOUT_S = (10, 60)  # (to connect, to wait for more data)
DOWNLOAD_RETRY_AFTER_S = 600  # after a failed download, do not try that file again for 10 minutes
RELEASE_FAILED_MESSAGE = (
    "We couldn't get the ready-made street file, so the streets are being downloaded the slower way instead. "
    "This may take a little longer."
)
_prebuilt_lock = threading.Lock()  # one file download at a time; a second visitor then finds the file already there
_release_failed_at: dict[str, float] = {}


def _could_cover(entry: ReleaseFile, lat: float, lon: float, radius_m: float, network_type: str) -> bool:
    """A quick guess from the published area, before downloading. (The file's own label decides for real.)"""
    if entry.network_type != network_type:
        return False
    left, bottom, right, top = _bbox(lat, lon, radius_m)
    have_left, have_bottom, have_right, have_top = _bbox(entry.lat, entry.lon, entry.radius_m + 100)  # + a little slack
    return have_left <= left and have_bottom <= bottom and right <= have_right and top <= have_top


def _download_file(url: str, path: Path) -> None:
    """Fetch ``url`` into ``path``, a piece at a time (so the file never sits whole in memory), and check it opens."""
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_S) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        with gzip.open(tmp_path, "rb") as f:  # a quick check that it is one of our files, not an error page
            label = pickle.load(f)
        if not isinstance(label, dict) or "radius_m" not in label:
            raise ValueError("not a street file")
        tmp_path.replace(path)  # atomic: a half-finished download is never mistaken for a good file
    finally:
        tmp_path.unlink(missing_ok=True)


def fetch_release_files(
    lat: float,
    lon: float,
    activity: str,
    radius_m: float,
    folder: Path,
    files: tuple[ReleaseFile, ...] | None = None,
    on_notice: Callable[[str], None] | None = None,
) -> bool:
    """Download the published file(s) that could cover the request into ``folder``. True if one was fetched.

    Files already in ``folder`` are not fetched again. A download that fails (no internet, GitHub down, a
    bad file) never raises: it tells ``on_notice`` in plain words and returns False, so the caller carries on
    with the live download. A file that failed is not retried for ``DOWNLOAD_RETRY_AFTER_S`` seconds, so
    people are not made to wait for a timeout on every request.
    """
    network_type = _network_type(activity)
    candidates = [e for e in (RELEASE_FILES if files is None else files) if _could_cover(e, lat, lon, radius_m, network_type)]
    fetched = False
    with _prebuilt_lock:
        for entry in candidates:
            path = Path(folder) / entry.name
            if path.exists():
                continue  # already here (perhaps fetched while we waited for the lock)
            if time.monotonic() - _release_failed_at.get(entry.url, -DOWNLOAD_RETRY_AFTER_S) < DOWNLOAD_RETRY_AFTER_S:
                continue
            try:
                Path(folder).mkdir(parents=True, exist_ok=True)
                _download_file(entry.url, path)
                fetched = True
            except Exception as error:  # no internet, a 404, a full disk, a bad file ... all mean "build it ourselves"
                _release_failed_at[entry.url] = time.monotonic()
                log.warning("Could not download %s: %s: %s", entry.url, type(error).__name__, error)
                if on_notice:
                    on_notice(RELEASE_FAILED_MESSAGE)
    return fetched


def _prebuilt_folders(preferred: Path) -> tuple[list[Path], Path]:
    """(folders to look in, folder to save downloads in). A folder we cannot write to falls back to a temporary one.

    Nothing is created here: the folder is only made when a file is actually saved into it.
    """
    preferred = Path(preferred)
    nearest = preferred
    while not nearest.exists() and nearest != nearest.parent:
        nearest = nearest.parent  # the closest folder that exists is the one we would have to write into
    if os.access(nearest, os.W_OK):
        return [preferred], preferred
    spare = Path(tempfile.gettempdir()) / "land_grab_prebuilt"
    return [preferred, spare], spare


def get_network(
    lat: float,
    lon: float,
    activity: str,
    radius_m: float,
    cache_dir: Path = CACHE_DIR,
    on_busy: Callable[[], None] | None = None,
    prebuilt_dir: Path | None = None,
    on_notice: Callable[[str], None] | None = None,
    release_files: tuple[ReleaseFile, ...] | None = None,
    on_slow: Callable[[], None] | None = None,
) -> nx.MultiDiGraph:
    """Return the walk/bike street network around a point, without motorways or trunk roads.

    "run" and "walk" use OSMnx's "walk" network; "ride" uses "bike". The streets come from the first of:

    1. the local pickle cache under ``cache_dir`` (default ``./cache``) from an earlier download of the
       same rounded lat/lon, radius and activity;
    2. a ready-made file in ``prebuilt_dir`` (default ``./prebuilt``) whose area covers the request
       (see ``find_prebuilt``); a damaged file is skipped, with a friendly message to ``on_notice``;
    3. the same kind of file downloaded from the project's GitHub Release into ``prebuilt_dir``
       (``fetch_release_files``; ``release_files`` replaces the list of published files, mainly for tests);
       if that fails the next step takes over;
    4. a live download (``download_graph``), which can take minutes: ``on_slow`` is called just before it
       starts, and only then (never when a saved or ready-made file was used): a busy map server is retried and other servers are tried,
       with ``on_busy`` called before each wait. A failed download is never saved: the file is only
       written once the streets have arrived, so the next attempt starts fresh.
    """
    network_type = _network_type(activity)
    path = cache_path(lat, lon, activity, radius_m, cache_dir)

    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)

    folders, save_folder = _prebuilt_folders(PREBUILT_DIR if prebuilt_dir is None else prebuilt_dir)
    for attempt in range(2):  # the second look happens only after a successful download
        for folder in folders:
            ready_made = find_prebuilt(lat, lon, activity, radius_m, folder, on_notice)
            if ready_made is not None:
                return ready_made
        if attempt == 0 and not fetch_release_files(lat, lon, activity, radius_m, save_folder, release_files, on_notice):
            break

    if on_slow:
        on_slow()  # nothing saved or ready-made covers this place: warn that the next part is slow
    graph = download_graph(lat, lon, network_type, radius_m, on_busy)
    graph = remove_excluded_roads(graph)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)  # atomic, so a crash never leaves a half-written cache file
    return graph
