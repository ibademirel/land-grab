import networkx as nx
import pytest

from land_grab import network
from land_grab.network import cache_key, cache_path, get_network, remove_excluded_roads


def _graph():
    g = nx.MultiDiGraph()
    g.add_edge(1, 2, highway="residential")
    g.add_edge(2, 3, highway="motorway")
    g.add_edge(3, 4, highway="trunk_link")
    g.add_edge(4, 5, highway=["footway", "trunk"])  # merged edge with a list tag
    g.add_edge(5, 6, highway="path")
    g.add_edge(7, 8, highway="motorway")  # only reachable via a motorway
    return g


def test_cache_key_is_stable():
    assert cache_key(51.501009, -0.141588, "run", 3000) == "run_51.5010_-0.1416_3000"
    assert cache_key(51.501009, -0.141588, "run", 3000) == cache_key(51.501009, -0.141588, "run", 3000)


def test_cache_key_ignores_tiny_coordinate_differences():
    assert cache_key(51.50100, -0.14160, "walk", 3000) == cache_key(51.501004, -0.141596, "walk", 3000)


def test_cache_key_normalises_activity_and_radius():
    assert cache_key(51.5, -0.14, " RUN ", 3000.4) == cache_key(51.5, -0.14, "run", 3000)


def test_cache_key_differs_by_activity_radius_and_location():
    base = cache_key(51.5010, -0.1416, "run", 3000)
    assert cache_key(51.5010, -0.1416, "ride", 3000) != base
    assert cache_key(51.5010, -0.1416, "run", 3001) != base
    assert cache_key(51.5020, -0.1416, "run", 3000) != base
    assert cache_key(51.5010, -0.1426, "run", 3000) != base


def test_cache_path_is_inside_cache_dir(tmp_path):
    path = cache_path(51.5010, -0.1416, "run", 3000, tmp_path)
    assert path.parent == tmp_path
    assert path.suffix == ".pkl"


def test_unknown_activity_raises_value_error():
    with pytest.raises(ValueError, match="Unknown activity"):
        cache_key(51.5, -0.14, "swim", 1000)


def test_remove_excluded_roads_drops_motorway_and_trunk_edges():
    cleaned = remove_excluded_roads(_graph())
    assert set(cleaned.edges(keys=False)) == {(1, 2), (5, 6)}


def test_remove_excluded_roads_drops_stranded_nodes_and_keeps_input_intact():
    original = _graph()
    cleaned = remove_excluded_roads(original)
    assert set(cleaned.nodes) == {1, 2, 5, 6}
    assert original.number_of_edges() == 6


@pytest.mark.parametrize(
    "activity, expected_type", [("run", "walk"), ("walk", "walk"), ("ride", "bike")]
)
def test_get_network_uses_right_network_type_and_caches(monkeypatch, tmp_path, activity, expected_type):
    calls = []

    def fake_graph_from_point(center, dist, network_type, custom_filter):
        calls.append((center, dist, network_type))
        assert custom_filter == network.street_filter(expected_type)  # the shared filter is what is sent
        return _graph()

    monkeypatch.setattr(network.ox, "graph_from_point", fake_graph_from_point)

    first = get_network(51.5010, -0.1416, activity, 3000, cache_dir=tmp_path)
    second = get_network(51.5010, -0.1416, activity, 3000, cache_dir=tmp_path)

    assert calls == [((51.5010, -0.1416), 3000, expected_type)]  # downloaded once only
    assert set(first.edges(keys=False)) == {(1, 2), (5, 6)}  # motorway/trunk removed
    assert set(second.edges(keys=False)) == set(first.edges(keys=False))
    assert len(list(tmp_path.glob("*.pkl"))) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_get_network_rejects_unknown_activity_before_downloading(monkeypatch, tmp_path):
    monkeypatch.setattr(
        network.ox, "graph_from_point", lambda *a, **k: pytest.fail("should not download")
    )
    with pytest.raises(ValueError):
        get_network(51.5, -0.14, "swim", 1000, cache_dir=tmp_path)


def test_a_missing_cache_folder_is_created(tmp_path):
    folder = tmp_path / "does" / "not" / "exist"
    assert network.usable_cache_dir(folder) == folder and folder.is_dir()


def test_an_unwritable_cache_folder_falls_back_to_a_temporary_one(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    chosen = network.usable_cache_dir(blocker / "cache")  # cannot make a folder inside a file
    assert chosen.is_dir() and chosen != blocker / "cache"


def test_the_cache_sits_next_to_the_project_not_where_the_program_was_started(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert network.CACHE_DIR.is_absolute() and network.CACHE_DIR.name == "cache"
    assert network.CACHE_DIR.parent == network.PROJECT_DIR
    assert network.ox.settings.cache_folder == str(network.CACHE_DIR)


# --- a busy map server --------------------------------------------------------------------
# A fake Overpass server: requests really go through the requests library (so OSMnx's own code and
# our "busy" check run for real), but the network is replaced by a script of answers per web address.

import requests  # noqa: E402

GOOD_ANSWER = (
    b'{"elements":[{"type":"node","id":1,"lat":51.5,"lon":-0.14},{"type":"node","id":2,"lat":51.501,"lon":-0.14},'
    b'{"type":"way","id":10,"nodes":[1,2],"tags":{"highway":"residential"}}]}'
)
AT_CAPACITY = b"<html><body>The server is probably too busy to handle your request. at capacity</body></html>"


@pytest.fixture
def fake_servers(monkeypatch, tmp_path):
    """Script the answers: ``script["overpass-api.de"] = [502, 502, 200]`` (the last one repeats)."""
    script, asked, sleeps = {}, [], []

    def send(self, request, **kwargs):
        host = request.url.split("/")[2]
        asked.append(host)
        answers = script.get(host, [200])
        status = answers[min(len([h for h in asked if h == host]) - 1, len(answers) - 1)]
        if status == "down":
            raise requests.ConnectionError(f"{host} is not answering")
        response = requests.Response()
        response.status_code = status
        response.url = request.url
        response._content = GOOD_ANSWER if status == 200 else AT_CAPACITY
        response.reason = "OK" if status == 200 else "Bad Gateway"
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    # No real waiting in tests. OSMnx sleeps 0 s before every request (its own rate-limit pause is switched
    # off below), so only real waits are written down: those are ours.
    monkeypatch.setattr(network.time, "sleep", lambda seconds: sleeps.append(seconds) if seconds else None)
    monkeypatch.setattr(network.ox._http, "_config_dns", lambda url: None)
    monkeypatch.setattr(network.ox.settings, "overpass_rate_limit", False)
    monkeypatch.setattr(network.ox.settings, "use_cache", False)
    script["asked"], script["sleeps"] = asked, sleeps
    return script


def _get(tmp_path, on_busy=None):
    return get_network(51.5, -0.14, "run", 500, cache_dir=tmp_path, on_busy=on_busy)


def test_a_server_that_works_is_used_once_with_no_waiting(fake_servers, tmp_path):
    messages = []
    graph = _get(tmp_path, lambda: messages.append("busy"))
    assert graph.number_of_edges() >= 1
    assert fake_servers["asked"] == ["overpass-api.de"] and fake_servers["sleeps"] == [] and messages == []


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_a_busy_answer_is_retried_after_5_then_15_seconds_and_then_succeeds(fake_servers, tmp_path, status):
    fake_servers["overpass-api.de"] = [status, status, 200]
    messages = []
    graph = _get(tmp_path, lambda: messages.append(network.BUSY_MESSAGE))

    assert graph.number_of_edges() >= 1
    assert fake_servers["asked"] == ["overpass-api.de"] * 3  # the same server, three tries in all
    assert fake_servers["sleeps"] == [5, 15]
    assert messages == [network.BUSY_MESSAGE] * 2  # told the user before each wait
    assert len(list(tmp_path.glob("*.pkl"))) == 1  # and the good answer was saved


def test_the_waits_grow_5_15_30_before_the_other_servers_are_tried(fake_servers, tmp_path):
    fake_servers["overpass-api.de"] = [502]  # always busy
    fake_servers["overpass.private.coffee"] = [200]
    graph = _get(tmp_path)

    assert graph.number_of_edges() >= 1
    assert fake_servers["asked"] == ["overpass-api.de"] * 4 + ["overpass.private.coffee"]  # 1 try + 3 retries
    assert fake_servers["sleeps"] == [5, 15, 30]


def test_the_servers_are_tried_one_after_another(fake_servers, tmp_path):
    for host in ("overpass-api.de", "overpass.private.coffee"):
        fake_servers[host] = [503]
    fake_servers["maps.mail.ru"] = [200]
    _get(tmp_path)
    assert fake_servers["asked"][-1] == "maps.mail.ru"
    assert fake_servers["asked"].count("overpass.private.coffee") == 1  # the others get one try each


def test_a_server_that_is_down_is_skipped_without_waiting(fake_servers, tmp_path):
    fake_servers["overpass-api.de"] = ["down"]
    _get(tmp_path)
    assert fake_servers["asked"] == ["overpass-api.de", "overpass.private.coffee"] and fake_servers["sleeps"] == []


def test_when_every_server_is_busy_the_error_is_raised_and_nothing_is_saved(fake_servers, tmp_path):
    for host in ("overpass-api.de", "overpass.private.coffee", "maps.mail.ru"):
        fake_servers[host] = [502]
    with pytest.raises(network.MapServiceBusyError, match="502"):
        _get(tmp_path)
    assert list(tmp_path.iterdir()) == []  # a failed download is never cached
    assert len(fake_servers["asked"]) == 4 + 1 + 1


def test_a_failure_does_not_leave_a_cache_file_and_the_next_try_downloads_again(fake_servers, tmp_path):
    fake_servers["overpass-api.de"] = [502, 502, 502, 502, 200]
    fake_servers["overpass.private.coffee"] = [502]
    fake_servers["maps.mail.ru"] = [502]
    with pytest.raises(network.MapServiceBusyError):
        _get(tmp_path)
    assert list(tmp_path.glob("*")) == []
    assert _get(tmp_path).number_of_edges() >= 1  # the first server has recovered
    assert len(list(tmp_path.glob("*.pkl"))) == 1


def test_osmnx_settings_are_put_back_after_a_download(fake_servers, tmp_path):
    before = (network.ox.settings.overpass_url, dict(network.ox.settings.requests_kwargs))
    fake_servers["overpass-api.de"] = [502]
    _get(tmp_path)
    assert (network.ox.settings.overpass_url, network.ox.settings.requests_kwargs) == before


def test_the_cached_streets_are_used_without_asking_any_server(fake_servers, tmp_path):
    _get(tmp_path)
    fake_servers["asked"].clear()
    _get(tmp_path)
    assert fake_servers["asked"] == []


def test_the_server_list_names_the_main_server_first_and_no_duplicates():
    assert network.OVERPASS_SERVERS[0] == "https://overpass-api.de/api"
    assert len(set(network.OVERPASS_SERVERS)) == len(network.OVERPASS_SERVERS) >= 2
    assert all(url.startswith("https://") and not url.endswith("/") for url in network.OVERPASS_SERVERS)
    assert network.RETRY_WAITS_S == (5, 15, 30)
    assert network.BUSY_STATUS_CODES == {429, 502, 503, 504}


# --- ready-made street files (prebuilt/) ---------------------------------------------------

import gzip  # noqa: E402

from land_grab.network import find_prebuilt, filter_fingerprint, prebuilt_name, save_prebuilt, street_filter  # noqa: E402

LAT, LON = 51.5, -0.14


def _streets(lat=LAT, lon=LON, step=0.002, size=5):
    """A tiny square street grid of size x size crossroads around (lat, lon), in the shape OSMnx makes."""
    g = nx.MultiDiGraph(crs="epsg:4326")
    half = size // 2
    for i in range(size):
        for j in range(size):
            g.add_node(i * size + j, x=lon + (j - half) * step, y=lat + (i - half) * step, street_count=4)
    for i in range(size):
        for j in range(size):
            for di, dj in ((0, 1), (1, 0)):
                a, b = i * size + j, (i + di) * size + (j + dj)
                if i + di < size and j + dj < size:
                    for u, v in ((a, b), (b, a)):
                        g.add_edge(u, v, 0, length=100.0, highway="residential", name="Some Street", osmid=7, oneway=False)
    return g


@pytest.fixture
def live(monkeypatch):
    """Counts live downloads and answers with the tiny grid."""
    calls = []

    def fake(center, dist, network_type, custom_filter):
        calls.append((center, dist, network_type))
        return _streets()

    monkeypatch.setattr(network.ox, "graph_from_point", fake)
    return calls


def _make_file(folder, radius_m=2000, activity="run", lat=LAT, lon=LON, area="test", graph=None):
    path = folder / prebuilt_name(area, radius_m, activity)
    save_prebuilt(graph if graph is not None else _streets(lat, lon), path, lat, lon, radius_m, activity, area)
    return path


def test_the_filter_asks_only_for_walkable_or_cyclable_streets_and_never_motorways():
    for kind in ("walk", "bike"):
        text = street_filter(kind)
        assert text.startswith('["highway"]') and '["highway"!~"motorway|motorway_link|trunk|trunk_link"]' in text
    assert "cycleway" in street_filter("walk").split("highway")[2]  # walkers skip cycle-only ways...
    assert "footway|" in street_filter("bike")  # ...and riders skip footpaths and steps
    with pytest.raises(ValueError):
        street_filter("drive")


def test_the_file_names_say_the_area_the_size_and_the_activity():
    assert prebuilt_name("london-se17", 10000, "run") == "london-se17-10km_run.pkl.gz"
    assert prebuilt_name("london-sw1a", 5000, "Ride") == "london-sw1a-5km_ride.pkl.gz"


def test_a_ready_made_file_is_used_when_it_covers_the_request_and_nothing_is_downloaded(live, tmp_path):
    _make_file(tmp_path, radius_m=2000)
    graph = get_network(LAT, LON, "run", 500, cache_dir=tmp_path / "cache", prebuilt_dir=tmp_path)
    assert live == []  # no live download
    assert graph.number_of_nodes() > 0 and graph.graph["crs"] == "epsg:4326"
    assert not list((tmp_path / "cache").glob("*.pkl"))  # and nothing new is written to the local cache


def test_a_ride_file_is_not_used_for_a_run_and_a_run_file_serves_a_walk(live, tmp_path):
    _make_file(tmp_path, activity="run")
    get_network(LAT, LON, "walk", 500, cache_dir=tmp_path / "c1", prebuilt_dir=tmp_path)
    assert live == []  # run and walk share the "walk" streets
    get_network(LAT, LON, "ride", 500, cache_dir=tmp_path / "c2", prebuilt_dir=tmp_path)
    assert len(live) == 1  # a run file has no cycle paths: ride goes live


def test_the_live_download_is_used_when_no_file_covers_the_request(live, tmp_path):
    _make_file(tmp_path, radius_m=500)
    get_network(LAT, LON, "run", 3000, cache_dir=tmp_path / "c1", prebuilt_dir=tmp_path)  # bigger than the file
    get_network(LAT + 0.5, LON, "run", 300, cache_dir=tmp_path / "c2", prebuilt_dir=tmp_path)  # somewhere else
    assert len(live) == 2


def test_an_empty_or_missing_prebuilt_folder_means_a_live_download(live, tmp_path):
    get_network(LAT, LON, "run", 500, cache_dir=tmp_path / "c1", prebuilt_dir=tmp_path)  # folder exists, empty
    get_network(LAT, LON, "run", 500, cache_dir=tmp_path / "c2", prebuilt_dir=tmp_path / "nope")
    assert len(live) == 2


def test_a_corrupt_file_falls_back_to_a_live_download_with_a_friendly_message(live, tmp_path):
    bad = tmp_path / prebuilt_name("broken", 2000, "run")
    bad.write_bytes(b"this is not a street file at all")
    notices = []
    graph = get_network(LAT, LON, "run", 500, cache_dir=tmp_path / "cache", prebuilt_dir=tmp_path, on_notice=notices.append)
    assert len(live) == 1 and graph.number_of_nodes() > 0
    assert notices and all(n == network.PREBUILT_DAMAGED_MESSAGE for n in notices)
    assert "damaged" in notices[0] and "Traceback" not in notices[0] and "gzip" not in notices[0]


def test_a_file_cut_short_in_the_streets_part_also_falls_back(live, tmp_path):
    path = _make_file(tmp_path, radius_m=2000)
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])  # a download that stopped half way
    notices = []
    get_network(LAT, LON, "run", 500, cache_dir=tmp_path / "cache", prebuilt_dir=tmp_path, on_notice=notices.append)
    assert len(live) == 1 and notices


def test_a_damaged_file_does_not_stop_a_good_one_next_to_it(live, tmp_path):
    (tmp_path / "aaa_broken.pkl.gz").write_bytes(b"junk")
    _make_file(tmp_path, radius_m=2000)
    notices = []
    graph = get_network(LAT, LON, "run", 500, cache_dir=tmp_path / "cache", prebuilt_dir=tmp_path, on_notice=notices.append)
    assert live == [] and graph.number_of_nodes() > 0 and notices


def test_a_file_made_with_other_street_types_is_ignored(live, tmp_path, monkeypatch):
    _make_file(tmp_path, radius_m=2000)
    monkeypatch.setitem(network._STREET_FILTERS, "walk", network._STREET_FILTERS["walk"] + '["surface"!~"mud"]')
    get_network(LAT, LON, "run", 500, cache_dir=tmp_path / "cache", prebuilt_dir=tmp_path)
    assert len(live) == 1  # the filter changed, so the old file is out of date
    assert filter_fingerprint("walk") != filter_fingerprint("bike")


def test_only_the_streets_inside_the_request_are_kept(tmp_path):
    big = _streets(size=9)  # about 1.1 km across
    path = _make_file(tmp_path, radius_m=1000, graph=big)
    small = find_prebuilt(LAT, LON, "run", 250, tmp_path)
    full = find_prebuilt(LAT, LON, "run", 1000, tmp_path)
    assert 0 < small.number_of_nodes() < full.number_of_nodes() <= big.number_of_nodes()
    left, bottom, right, top = network._bbox(LAT, LON, 250)
    assert all(left <= d["x"] <= right and bottom <= d["y"] <= top for _, d in small.nodes(data=True))
    assert path.exists()


def test_the_saved_streets_keep_what_the_planner_needs_and_drop_the_rest(tmp_path):
    from shapely.geometry import LineString

    g = _streets()
    g.add_edge(0, 1, 1, length=50.0, highway=["residential", "service"], geometry=LineString([(-0.14, 51.5), (-0.139, 51.5)]))
    path = _make_file(tmp_path, graph=g)
    loaded = find_prebuilt(LAT, LON, "run", 500, tmp_path)
    edges = [d for _, _, d in loaded.edges(data=True)]
    assert all(set(d) <= {"length", "highway", "geometry"} for d in edges)  # names, ids, speeds are gone
    assert any("geometry" in d for d in edges) and any(isinstance(d["highway"], list) for d in edges)
    assert all(set(d) == {"x", "y"} for _, d in loaded.nodes(data=True))
    assert path.stat().st_size < 50_000


def test_the_label_can_be_read_without_the_streets(tmp_path):
    import pickle

    path = _make_file(tmp_path, radius_m=2000, area="london-se17")
    with gzip.open(path, "rb") as f:
        label = pickle.load(f)
    assert label["area"] == "london-se17" and label["radius_m"] == 2000 and label["network_type"] == "walk"
    assert label["nodes"] == 25 and label["edges"] > 0


def test_the_smallest_covering_file_is_used(tmp_path):
    _make_file(tmp_path, radius_m=3000, area="big", graph=_streets(size=9))
    _make_file(tmp_path, radius_m=1000, area="small", graph=_streets(size=5))
    graph = find_prebuilt(LAT, LON, "run", 300, tmp_path)
    assert graph.number_of_nodes() <= 25  # came from the small one (5 x 5 crossroads)


def test_the_local_cache_still_comes_first(live, tmp_path):
    get_network(LAT, LON, "run", 500, cache_dir=tmp_path / "cache", prebuilt_dir=tmp_path / "none")  # downloads, saves
    _make_file(tmp_path, radius_m=2000)
    get_network(LAT, LON, "run", 500, cache_dir=tmp_path / "cache", prebuilt_dir=tmp_path)
    assert len(live) == 1


def test_a_prebuilt_network_works_with_the_planner_helpers(tmp_path):
    import osmnx as ox

    _make_file(tmp_path, radius_m=2000)
    graph = find_prebuilt(LAT, LON, "run", 500, tmp_path)
    node = ox.distance.nearest_nodes(graph, X=LON, Y=LAT)  # needs the saved CRS
    assert node in graph


# --- downloading the ready-made files from the GitHub Release -------------------------------

from land_grab.network import RELEASE_FAILED_MESSAGE, ReleaseFile  # noqa: E402

REAL_RELEASE_FILES = network.RELEASE_FILES  # read now: the test set-up empties network.RELEASE_FILES so no test reaches GitHub
URL = "https://example.test/releases/london-test-2km_run.pkl.gz"
ENTRY = ReleaseFile(URL, "run", LAT, LON, 2000)


class _FakeResponse:
    def __init__(self, body=b"", status=200):
        self.body, self.status = body, status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(f"{self.status} error")

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), 7):  # tiny pieces, to show the file is written piece by piece
            yield self.body[i : i + 7]


@pytest.fixture
def github(monkeypatch, tmp_path):
    """A pretend GitHub: ``github.body`` is what it sends, ``github.error`` what goes wrong; ``github.asked`` the URLs."""
    made = tmp_path / "published"
    made.mkdir()
    published = _make_file(made, radius_m=2000)

    class Fake:
        body = published.read_bytes()
        error = None
        status = 200
        asked = []

    def get(url, **kwargs):
        Fake.asked.append(url)
        if Fake.error:
            raise Fake.error
        return _FakeResponse(Fake.body, Fake.status)

    monkeypatch.setattr(network.requests, "get", get)
    Fake.asked = []
    return Fake


def _fetch_get(tmp_path, notices=None, **kwargs):
    return get_network(
        LAT, LON, kwargs.pop("activity", "run"), kwargs.pop("radius", 500), cache_dir=tmp_path / "cache",
        prebuilt_dir=tmp_path / "prebuilt", release_files=kwargs.pop("files", (ENTRY,)),
        on_notice=notices.append if notices is not None else None,
    )


def test_the_published_file_is_downloaded_saved_and_used_instead_of_a_live_download(live, github, tmp_path):
    graph = _fetch_get(tmp_path)
    assert github.asked == [URL] and live == []  # fetched from GitHub, Overpass not asked
    assert (tmp_path / "prebuilt" / "london-test-2km_run.pkl.gz").exists()
    assert graph.number_of_nodes() > 0 and graph.graph["crs"] == "epsg:4326"
    assert not list((tmp_path / "prebuilt").glob("*.tmp"))


def test_a_file_already_on_this_computer_is_not_downloaded_again(live, github, tmp_path):
    _fetch_get(tmp_path)
    _fetch_get(tmp_path)
    get_network(LAT, LON, "run", 400, cache_dir=tmp_path / "c2", prebuilt_dir=tmp_path / "prebuilt", release_files=(ENTRY,))
    assert github.asked == [URL] and live == []  # once only, even for a different distance


@pytest.mark.parametrize(
    "problem",
    [
        {"error": requests.ConnectionError("no internet")},
        {"error": requests.Timeout("too slow")},
        {"status": 404},  # the file is not published (yet)
        {"body": b"<html>Not Found</html>"},  # an error page instead of a street file
    ],
)
def test_if_the_download_fails_the_streets_are_built_the_old_way_with_a_friendly_message(live, github, tmp_path, problem):
    for name, value in problem.items():
        setattr(github, name, value)
    notices = []
    graph = _fetch_get(tmp_path, notices)
    assert len(live) == 1 and graph.number_of_nodes() > 0  # the live download took over
    assert notices == [RELEASE_FAILED_MESSAGE]
    assert "couldn't" in notices[0] and "Traceback" not in notices[0]
    assert not list((tmp_path / "prebuilt").glob("*"))  # no half-finished file left behind


def test_after_a_failed_download_the_next_visitor_is_not_made_to_wait_again(live, github, tmp_path):
    github.error = requests.ConnectionError("no internet")
    _fetch_get(tmp_path)
    _fetch_get(tmp_path)
    assert github.asked == [URL]  # not tried a second time straight away
    assert len(live) == 1  # (the second request came from the local cache of the first live download)


def test_nothing_is_downloaded_for_an_area_or_activity_no_published_file_covers(live, github, tmp_path):
    get_network(LAT + 0.5, LON, "run", 300, cache_dir=tmp_path / "c1", prebuilt_dir=tmp_path / "p", release_files=(ENTRY,))
    get_network(LAT, LON, "run", 5000, cache_dir=tmp_path / "c2", prebuilt_dir=tmp_path / "p", release_files=(ENTRY,))  # too big
    get_network(LAT, LON, "ride", 300, cache_dir=tmp_path / "c3", prebuilt_dir=tmp_path / "p", release_files=(ENTRY,))  # run file
    assert github.asked == [] and len(live) == 3


def test_a_walk_can_use_a_published_run_file(live, github, tmp_path):
    _fetch_get(tmp_path, activity="walk")
    assert github.asked == [URL] and live == []


def test_only_the_file_that_matches_the_request_is_downloaded(live, github, tmp_path):
    far = ReleaseFile("https://example.test/far_run.pkl.gz", "run", LAT + 0.5, LON, 2000)
    ride = ReleaseFile("https://example.test/here_ride.pkl.gz", "ride", LAT, LON, 2000)
    _fetch_get(tmp_path, files=(far, ride, ENTRY))
    assert github.asked == [URL]


def test_a_published_file_that_turns_out_not_to_cover_the_request_falls_back_to_live(live, github, tmp_path):
    # Claimed to cover 2 km, but the file inside only holds 500 m: the label inside the file decides.
    small = tmp_path / "small"
    small.mkdir()
    github.body = _make_file(small, radius_m=500).read_bytes()
    _fetch_get(tmp_path, radius=1500)
    assert len(live) == 1


def test_the_published_links_are_the_three_release_files():
    base = "https://github.com/ibademirel/land-grab/releases/download/streets-v1/"
    assert sorted(e.url for e in REAL_RELEASE_FILES) == sorted(
        base + name
        for name in ("london-se17-10km_ride.pkl.gz", "london-se17-10km_run.pkl.gz", "london-sw1a-5km_ride.pkl.gz")
    )


def test_the_published_files_are_named_like_the_ones_the_build_script_makes():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("build_street_cache", Path(__file__).parent.parent / "scripts" / "build_street_cache.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    made = {prebuilt_name(name, radius, activity) for name, _, radius in script.AREAS for activity in script.ACTIVITIES}
    # every published file is one the script makes (the script also makes a SW1A run file, which is not published)
    assert {e.name for e in REAL_RELEASE_FILES} <= made
    assert all(e.radius_m == dict((n, r) for n, _, r in script.AREAS)[e.name.split("_")[0].rsplit("-", 1)[0]] for e in REAL_RELEASE_FILES)


def test_a_folder_we_cannot_write_to_uses_a_temporary_one(monkeypatch, tmp_path):
    monkeypatch.setattr(network.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    monkeypatch.setattr(network.os, "access", lambda path, mode: False)
    looked, save = network._prebuilt_folders(tmp_path / "prebuilt")
    assert save == tmp_path / "tmp" / "land_grab_prebuilt" and looked == [tmp_path / "prebuilt", save]
    assert not (tmp_path / "prebuilt").exists()  # looking never creates folders


# --- warning before a slow first-time load -------------------------------------------------


def test_the_slow_warning_comes_just_before_a_live_download_and_never_after_it(live, tmp_path):
    order = []
    get_network(LAT, LON, "run", 500, cache_dir=tmp_path / "c", prebuilt_dir=tmp_path / "p", on_slow=lambda: order.append(len(live)))
    assert order == [0] and len(live) == 1  # the warning was given when no download had started yet


def test_the_slow_warning_is_not_given_when_a_ready_made_file_is_used(live, tmp_path):
    _make_file(tmp_path, radius_m=2000)
    warned = []
    get_network(LAT, LON, "run", 500, cache_dir=tmp_path / "c", prebuilt_dir=tmp_path, on_slow=lambda: warned.append(1))
    assert warned == [] and live == []


def test_the_slow_warning_is_not_given_when_a_published_file_is_downloaded(live, github, tmp_path):
    warned = []
    get_network(
        LAT, LON, "run", 500, cache_dir=tmp_path / "c", prebuilt_dir=tmp_path / "p", release_files=(ENTRY,), on_slow=lambda: warned.append(1)
    )
    assert warned == [] and live == []


def test_the_slow_warning_is_not_given_when_the_local_cache_has_the_streets(live, tmp_path):
    get_network(LAT, LON, "run", 500, cache_dir=tmp_path / "c", prebuilt_dir=tmp_path / "p")
    warned = []
    get_network(LAT, LON, "run", 500, cache_dir=tmp_path / "c", prebuilt_dir=tmp_path / "p", on_slow=lambda: warned.append(1))
    assert warned == []


def test_the_slow_warning_is_given_if_the_published_download_fails_too(live, github, tmp_path):
    github.error = requests.ConnectionError("no internet")
    warned = []
    get_network(
        LAT, LON, "run", 500, cache_dir=tmp_path / "c", prebuilt_dir=tmp_path / "p", release_files=(ENTRY,), on_slow=lambda: warned.append(1)
    )
    assert warned == [1] and len(live) == 1


def test_the_slow_warning_wording():
    assert network.FIRST_TIME_MESSAGE == (
        "First time planning in this area. Loading the street map can take 1 to 3 minutes, "
        "so please keep this page open. Central and south London load instantly."
    )
