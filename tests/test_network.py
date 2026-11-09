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
