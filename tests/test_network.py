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

    def fake_graph_from_point(center, dist, network_type):
        calls.append((center, dist, network_type))
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
