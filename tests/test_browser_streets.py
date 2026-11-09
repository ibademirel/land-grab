"""The phone download: the question we send the phone, and turning its answer into streets."""

import base64
import gzip
import json
import shutil
import subprocess
from pathlib import Path

import networkx as nx
import osmnx as ox
import pytest
from osmnx import _overpass

from land_grab import browser_streets as bs
from land_grab import network, pipeline
from land_grab.browser_streets import NeedPhoneStreets, PhoneStreetsError, build_request, encode_reply, graph_from_reply

LAT, LON = 51.5000, -0.1000
RADIUS = 600.0
ROOT = Path(__file__).resolve().parent.parent


def fake_osm(extra_tags=True):
    """A small street grid around the start, as Overpass would answer (with tags the planner does not need)."""
    n = 9
    step = 0.0012
    elements, ids = [], {}
    for i in range(n):
        for j in range(n):
            ids[i, j] = 1000 + i * n + j
            elements.append({"type": "node", "id": ids[i, j], "lat": LAT + (i - 4) * step * 0.7, "lon": LON + (j - 4) * step})
    way_id = 1
    for i in range(n):
        tags = {"highway": "residential", "name": f"Row {i}"} if extra_tags else {"highway": "residential"}
        if i == 2:
            tags["oneway"] = "yes"
        elements.append({"type": "way", "id": way_id, "nodes": [ids[i, j] for j in range(n)], "tags": tags})
        way_id += 1
    for j in range(n):
        tags = {"highway": "footway" if j == 5 else "tertiary", "surface": "paving"} if extra_tags else {"highway": "tertiary"}
        elements.append({"type": "way", "id": way_id, "nodes": [ids[i, j] for i in range(n)], "tags": tags})
        way_id += 1
    elements.append({"type": "node", "id": 5, "lat": LAT, "lon": LON, "tags": {"highway": "crossing", "name": "x"}})
    return elements


def osmnx_graph(monkeypatch, network_type, elements, queries):
    def fake_request(data):
        queries.append(data["data"])
        return {"elements": elements}

    monkeypatch.setattr(_overpass, "_overpass_request", fake_request)
    return ox.graph_from_point((LAT, LON), dist=RADIUS, network_type=network_type, custom_filter=network.street_filter(network_type))


# --- the question asked ------------------------------------------------------------------------------


@pytest.mark.parametrize("network_type", ["walk", "bike"])
def test_the_phone_is_asked_exactly_what_the_server_download_asks(monkeypatch, network_type):
    asked = []
    osmnx_graph(monkeypatch, network_type, fake_osm(), asked)
    request = build_request(LAT, LON, network_type, RADIUS)
    assert list(request.queries) == asked and len(asked) == 1
    assert network.street_filter(network_type) in asked[0]  # the same street filter as the server download


def test_the_request_carries_the_servers_the_waits_and_a_new_id_each_time():
    a, b = build_request(LAT, LON, "walk", RADIUS), build_request(LAT, LON, "walk", RADIUS)
    assert a.request_id != b.request_id
    data = a.for_browser()
    json.dumps(data)  # plain data only
    assert data["servers"] == [server + "/interpreter" for server in network.OVERPASS_SERVERS]
    assert data["waits"] == list(network.RETRY_WAITS_S) and 429 in data["busyStatuses"] and 504 in data["busyStatuses"]
    assert data["maxPayloadChars"] == bs.MAX_PAYLOAD_CHARS and data["format"] == bs.REPLY_FORMAT


def test_a_big_area_is_asked_in_several_pieces():
    request = build_request(LAT, LON, "bike", 40_000)  # Overpass limits one question to about 2500 km²
    assert len(request.queries) > 1
    assert all(network.street_filter("bike") in query for query in request.queries)


# --- the answer ------------------------------------------------------------------------------------------


def _same(a: nx.MultiDiGraph, b: nx.MultiDiGraph):
    assert set(a.nodes) == set(b.nodes)
    for node in a.nodes:
        assert (a.nodes[node]["x"], a.nodes[node]["y"]) == (b.nodes[node]["x"], b.nodes[node]["y"])
        assert a.nodes[node].get("street_count") == b.nodes[node].get("street_count")
    assert sorted(a.edges(keys=True)) == sorted(b.edges(keys=True))
    for u, v, key, data in a.edges(keys=True, data=True):
        other = b.edges[u, v, key]
        assert other["length"] == pytest.approx(data["length"]) and other["highway"] == data["highway"]
        assert other["oneway"] == data["oneway"]
        assert (data.get("geometry") is None) == (other.get("geometry") is None)


@pytest.mark.parametrize("network_type", ["walk", "bike"])
@pytest.mark.parametrize("compress", [True, False])
def test_the_streets_from_the_phone_are_the_same_as_the_streets_osmnx_downloads(monkeypatch, network_type, compress):
    elements = fake_osm()
    expected = osmnx_graph(monkeypatch, network_type, elements, [])
    reply = encode_reply([elements], compress=compress)
    got = graph_from_reply(reply, LAT, LON, network_type, RADIUS)
    assert got.number_of_edges() > 50
    _same(expected, got)
    assert got.graph["crs"] == expected.graph["crs"]


def test_walking_streets_are_two_way_and_bike_streets_keep_one_way(monkeypatch):
    elements = fake_osm()
    walk = graph_from_reply(encode_reply([elements]), LAT, LON, "walk", RADIUS)
    bike = graph_from_reply(encode_reply([elements]), LAT, LON, "bike", RADIUS)
    assert not any(d["oneway"] for *_, d in walk.edges(data=True))
    assert any(d["oneway"] for *_, d in bike.edges(data=True))


def test_several_pieces_are_merged_without_doubling_streets(monkeypatch):
    elements = fake_osm()
    once = graph_from_reply(encode_reply([elements]), LAT, LON, "walk", RADIUS)
    twice = graph_from_reply(encode_reply([elements, elements]), LAT, LON, "walk", RADIUS)
    _same(once, twice)


# --- bad answers ---------------------------------------------------------------------------------------


def _payload(obj, encoding="json"):
    text = json.dumps(obj)
    if encoding == "gzip-base64":
        return {"ok": True, "encoding": encoding, "payload": base64.b64encode(gzip.compress(text.encode())).decode()}
    return {"ok": True, "encoding": encoding, "payload": text}


def _good():
    return json.loads(encode_reply([fake_osm()], compress=False)["payload"])


@pytest.mark.parametrize(
    "reply, fragment",
    [
        (None, "no answer"),
        ({"ok": False, "error": "Your phone seems to be offline."}, "offline"),
        ({"ok": False}, "could not download"),
        ({"ok": True, "encoding": "json", "payload": ""}, "empty"),
        ({"ok": True, "encoding": "json", "payload": "not json"}, "could not be read"),
        ({"ok": True, "encoding": "gzip-base64", "payload": "!!!"}, "could not be read"),
        ({"ok": True, "encoding": "gzip-base64", "payload": base64.b64encode(b"plain").decode()}, "could not be read"),
        ({"ok": True, "encoding": "zip", "payload": "x"}, "don't know"),
        ({"ok": True, "encoding": "json", "payload": "x" * (bs.MAX_PAYLOAD_CHARS + 1)}, "too big"),
        (_payload([1, 2]), "layout"),
        (_payload({"v": 99, "nodes": [], "ways": []}), "layout"),
        (_payload({"v": 1, "nodes": [], "ways": []}), "any streets"),
        (_payload({"v": 1}), "missing"),
    ],
)
def test_a_failed_or_unreadable_answer_is_a_clear_error(reply, fragment):
    with pytest.raises(PhoneStreetsError, match=fragment):
        graph_from_reply(reply, LAT, LON, "walk", RADIUS)


def _broken(change):
    data = _good()
    change(data)
    return _payload(data)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["nodes"].append(["x", 1, 2]),
        lambda d: d["nodes"].append([7, "a", 2]),
        lambda d: d["nodes"].append([7, 95.0, 2]),
        lambda d: d["nodes"].append([7, float("nan"), 2]),
        lambda d: d["ways"].append([9, "residential", "", "", [1000, 424242]]),  # uses a point that was not sent
        lambda d: d["ways"].append([9, "", "", "", [1000, 1001]]),  # no street type
        lambda d: d["ways"].append([9, "residential", "", "", [1000]]),  # one point only
        lambda d: d["ways"].append([9, 5, "", "", [1000, 1001]]),
        lambda d: d["ways"].append("street"),
    ],
)
def test_rubbish_rows_are_refused_not_trusted(change):
    with pytest.raises(PhoneStreetsError):
        graph_from_reply(_broken(change), LAT, LON, "walk", RADIUS)


def test_streets_for_the_wrong_place_are_refused():
    with pytest.raises(PhoneStreetsError, match="could not be used"):
        graph_from_reply(encode_reply([fake_osm()]), LAT + 1.0, LON, "walk", RADIUS)


def test_a_gzip_bomb_is_stopped(monkeypatch):
    monkeypatch.setattr(bs, "MAX_JSON_BYTES", 1000)
    with pytest.raises(PhoneStreetsError, match="too big"):
        graph_from_reply(_payload({"v": 1, "nodes": [], "ways": [], "pad": "a" * 5000}, "gzip-base64"), LAT, LON, "walk", RADIUS)


# --- get_network: the order of things ------------------------------------------------------------------


@pytest.fixture
def no_server(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("must not download on the server")

    monkeypatch.setattr(network, "download_graph", refuse)


def test_nothing_local_and_ask_phone_stops_with_a_request(tmp_path, no_server):
    with pytest.raises(NeedPhoneStreets) as caught:
        network.get_network(LAT, LON, "run", RADIUS, cache_dir=tmp_path, prebuilt_dir=tmp_path / "none", ask_phone=True)
    request = caught.value.request
    assert request.network_type == "walk" and (request.lat, request.lon, request.radius_m) == (LAT, LON, RADIUS)


def test_the_saved_cache_beats_the_phone(tmp_path, no_server):
    graph = nx.MultiDiGraph()
    path = network.cache_path(LAT, LON, "run", RADIUS, tmp_path)
    import pickle

    path.write_bytes(pickle.dumps(graph))
    assert network.get_network(LAT, LON, "run", RADIUS, cache_dir=tmp_path, ask_phone=True).number_of_nodes() == 0


def test_a_ready_made_file_beats_the_phone(tmp_path, no_server):
    source = ox.graph_from_xml  # noqa: F841 - only to show we do not use the network
    graph = graph_from_reply(encode_reply([fake_osm()]), LAT, LON, "walk", RADIUS)
    network.save_prebuilt(graph, tmp_path / "pre" / "x.pkl.gz", LAT, LON, RADIUS, "run", area="test")
    got = network.get_network(LAT, LON, "run", RADIUS, cache_dir=tmp_path / "c", prebuilt_dir=tmp_path / "pre", ask_phone=True)
    assert got.number_of_edges() == graph.number_of_edges()


def test_the_phones_streets_are_used_and_saved_for_next_time(tmp_path, no_server):
    notes = []
    reply = encode_reply([fake_osm()])
    graph = network.get_network(
        LAT, LON, "run", RADIUS, cache_dir=tmp_path, prebuilt_dir=tmp_path / "none", phone_reply=reply, on_phone_failed=notes.append
    )
    assert graph.number_of_edges() > 50 and notes == []
    assert network.cache_path(LAT, LON, "run", RADIUS, tmp_path).exists()
    again = network.get_network(LAT, LON, "run", RADIUS, cache_dir=tmp_path, ask_phone=True)  # instant, no phone
    assert again.number_of_edges() == graph.number_of_edges()


def test_motorways_from_the_phone_are_removed_like_for_a_server_download(tmp_path, no_server):
    elements = fake_osm()
    ids = [e["id"] for e in elements if e["type"] == "node"][:2]
    elements.append({"type": "way", "id": 999, "nodes": ids, "tags": {"highway": "motorway"}})
    graph = network.get_network(
        LAT, LON, "run", RADIUS, cache_dir=tmp_path, prebuilt_dir=tmp_path / "none", phone_reply=encode_reply([elements])
    )
    assert not any(network._is_excluded(d["highway"]) for *_, d in graph.edges(data=True))


def test_when_the_phone_fails_the_server_takes_over_and_we_are_told(tmp_path, monkeypatch):
    notes, slow, served = [], [], nx.MultiDiGraph()
    served.add_edge(1, 2, highway="residential", length=5.0)
    monkeypatch.setattr(network, "download_graph", lambda *a, **k: served)
    graph = network.get_network(
        LAT, LON, "run", RADIUS, cache_dir=tmp_path, prebuilt_dir=tmp_path / "none",
        phone_reply={"ok": False, "error": "Your phone seems to be offline."}, on_phone_failed=notes.append, on_slow=lambda: slow.append(1),
    )
    assert graph.number_of_edges() == 1 and len(notes) == 1 and "offline" in notes[0] and slow == [1]


def test_without_ask_phone_nothing_changes(tmp_path, monkeypatch):
    served = nx.MultiDiGraph()
    served.add_edge(1, 2, highway="residential", length=5.0)
    monkeypatch.setattr(network, "download_graph", lambda *a, **k: served)
    got = network.get_network(LAT, LON, "run", RADIUS, cache_dir=tmp_path, prebuilt_dir=tmp_path / "none")
    assert got.number_of_edges() == 1


def test_the_command_line_never_asks_a_phone():
    import inspect

    from land_grab import cli

    assert "ask_phone" not in inspect.getsource(cli)


# --- the pipeline and the web layer pass it on -----------------------------------------------------------


def test_the_phone_failed_message_is_a_progress_stage(tmp_path, monkeypatch):
    from test_optimiser import _grid

    graph = _grid()
    seen = {}

    def fake_get_network(lat, lon, activity, radius_m, on_busy=None, on_notice=None, on_slow=None, **extra):
        seen.update(extra)
        extra["on_phone_failed"]("Your phone seems to be offline.")
        return graph

    monkeypatch.setattr(pipeline, "postcode_to_latlon", lambda postcode: (LAT, LON))
    monkeypatch.setattr(pipeline, "get_network", fake_get_network)
    monkeypatch.setattr(pipeline.ox.distance, "nearest_nodes", lambda graph, X, Y: (2, 1))
    stages = []
    pipeline.plan_loop("SW1A 1AA", 4.0, "walk", planner_only=True, on_stage=stages.append, ask_phone=True, phone_reply={"ok": False})
    assert pipeline.STAGE_PHONE_FAILED in stages and seen["phone_reply"] == {"ok": False} and seen["ask_phone"] is True
    assert pipeline.STAGE_PHONE_FAILED == "Your phone couldn't download the street map. Trying our server instead…"


def test_the_command_line_call_to_get_network_is_unchanged(monkeypatch):
    from test_optimiser import _grid

    seen = {}

    def fake_get_network(lat, lon, activity, radius_m, **kwargs):
        seen.update(kwargs)
        return _grid()

    monkeypatch.setattr(pipeline, "postcode_to_latlon", lambda postcode: (LAT, LON))
    monkeypatch.setattr(pipeline, "get_network", fake_get_network)
    monkeypatch.setattr(pipeline.ox.distance, "nearest_nodes", lambda graph, X, Y: (2, 1))
    pipeline.plan_loop("SW1A 1AA", 4.0, "walk", planner_only=True)
    assert "ask_phone" not in seen and "phone_reply" not in seen


# --- the JavaScript helper ---------------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_the_javascript_tests_pass():
    done = subprocess.run(["node", "--test", str(ROOT / "tests" / "js" / "phone_streets.test.mjs")], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]


def test_the_javascript_and_python_agree_on_the_reply_format():
    source = (ROOT / "land_grab" / "phone_streets.mjs").read_text()
    assert f"REPLY_FORMAT = {bs.REPLY_FORMAT};" in source
