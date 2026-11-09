"""Download street networks from OpenStreetMap (via OSMnx) with a local pickle cache."""

import os
import pickle
import tempfile
from pathlib import Path

import networkx as nx
import osmnx as ox

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
) -> nx.MultiDiGraph:
    """Return the walk/bike street network around a point, without motorways or trunk roads.

    "run" and "walk" use OSMnx's "walk" network; "ride" uses "bike". Results are
    cached as a pickle under ``cache_dir`` (default ``./cache``), so repeated
    calls for the same rounded lat/lon, radius and activity don't download again.
    """
    network_type = _network_type(activity)
    path = cache_path(lat, lon, activity, radius_m, cache_dir)

    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)

    graph = ox.graph_from_point((lat, lon), dist=radius_m, network_type=network_type)
    graph = remove_excluded_roads(graph)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
    with tmp_path.open("wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)  # atomic, so a crash never leaves a half-written cache file
    return graph
