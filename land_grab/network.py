"""Download street networks from OpenStreetMap (via OSMnx) with a local pickle cache."""

import os
import pickle
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import networkx as nx
import osmnx as ox
import requests

PROJECT_DIR = Path(__file__).resolve().parent.parent


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
                        return ox.graph_from_point((lat, lon), dist=radius_m, network_type=network_type)
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


def get_network(
    lat: float,
    lon: float,
    activity: str,
    radius_m: float,
    cache_dir: Path = CACHE_DIR,
    on_busy: Callable[[], None] | None = None,
) -> nx.MultiDiGraph:
    """Return the walk/bike street network around a point, without motorways or trunk roads.

    "run" and "walk" use OSMnx's "walk" network; "ride" uses "bike". Results are
    cached as a pickle under ``cache_dir`` (default ``./cache``), so repeated
    calls for the same rounded lat/lon, radius and activity don't download again.
    A busy map server is retried and other servers are tried (``download_graph``), with ``on_busy``
    called before each wait. A failed download is never saved: the file is only written once the
    streets have arrived, so the next attempt starts fresh.
    """
    network_type = _network_type(activity)
    path = cache_path(lat, lon, activity, radius_m, cache_dir)

    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)

    graph = download_graph(lat, lon, network_type, radius_m, on_busy)
    graph = remove_excluded_roads(graph)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)  # atomic, so a crash never leaves a half-written cache file
    return graph
