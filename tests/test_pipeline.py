import gpxpy
import pytest

from land_grab import pipeline
from land_grab.pipeline import (
    MAX_RADIUS_M,
    InvalidPostcodeError,
    gpx_text,
    map_html,
    network_radius_m,
    plan_loop,
)
from land_grab.hexgrid import _TO_WGS84, InvalidCellError, neighbours
from test_optimiser import _E0, _N0, HOME_CELL, _grid

_LON, _LAT = _TO_WGS84.transform(_E0 - 1800, _N0 - 900)


@pytest.fixture
def stubbed(monkeypatch):
    graph = _grid()
    monkeypatch.setattr(pipeline, "postcode_to_latlon", lambda postcode: (_LAT, _LON))
    monkeypatch.setattr(pipeline, "get_network", lambda *a, **k: graph)
    monkeypatch.setattr(pipeline.ox.distance, "nearest_nodes", lambda graph, X, Y: (2, 1))


def test_plan_loop_returns_everything_the_app_shows(stubbed):
    messages = []
    loop = plan_loop("SW1A 1AA", 4, "walk", n_candidates=20, on_status=messages.append)

    assert loop.start == (_LAT, _LON)
    assert loop.length_m == pytest.approx(4000, rel=0.05)
    assert len(loop.safe_hexes) >= 1
    assert loop.n_points == len(loop.route_latlons) > 2
    assert loop.route_latlons[0] == loop.route_latlons[-1]
    assert loop.gpx_name == "Land Grab SW1A 1AA 4 km walk"
    assert messages[0].startswith("start: ") and messages[1].startswith("network radius: ")


def test_map_html_and_gpx_text_are_ready_to_embed_and_download(stubbed):
    loop = plan_loop("SW1A 1AA", 4, "run", n_candidates=20)

    assert "leaflet" in map_html(loop).lower()
    gpx = gpxpy.parse(gpx_text(loop))
    assert len(gpx.tracks) == 1 and len(gpx.tracks[0].segments) == 1
    assert len(gpx.tracks[0].segments[0].points) == loop.n_points
    assert gpx.tracks[0].name == loop.gpx_name


def test_invalid_postcode_raises_a_specific_error(monkeypatch):
    def not_found(postcode):
        raise ValueError(f"Postcode not found: {postcode!r}")

    monkeypatch.setattr(pipeline, "postcode_to_latlon", not_found)
    with pytest.raises(InvalidPostcodeError):
        plan_loop("ZZ99 9ZZ", 5)
    with pytest.raises(ValueError):  # it is still a ValueError, so the CLI reports it too
        plan_loop("ZZ99 9ZZ", 5)


def test_bad_distance_is_rejected_before_any_lookup(monkeypatch):
    def fail(postcode):
        raise AssertionError("should not look up the postcode")

    monkeypatch.setattr(pipeline, "postcode_to_latlon", fail)
    with pytest.raises(ValueError):
        plan_loop("SW1A 1AA", 0)


def test_radius_cap_is_flagged(stubbed):
    assert network_radius_m(40) == MAX_RADIUS_M


# --- planner only and progress stages (the web app) ---------------------------------------


def test_planner_only_skips_the_random_search_and_reports_plain_stages(stubbed, monkeypatch):
    from land_grab import planner

    monkeypatch.setattr(planner, "optimise_route", lambda *a, **k: pytest.fail("random search must not run"))
    stages = []
    loop = plan_loop("SW1A 1AA", 4, "walk", planner_only=True, on_stage=stages.append)

    assert stages == ["Finding your postcode", "Loading the streets", "Choosing hexagons", "Joining them up"]
    assert loop.choice.chosen is loop.choice.planner
    assert loop.choice.random.error == "skipped"
    assert loop.length_m == pytest.approx(4000, rel=0.05)
    assert len(loop.safe_hexes) >= 1


def test_the_default_still_runs_both_searches(stubbed):
    loop = plan_loop("SW1A 1AA", 4, "walk", n_candidates=20)
    assert loop.choice.planner.route is not None and loop.choice.random.route is not None


def test_planner_only_is_not_used_by_the_cli(stubbed, monkeypatch):
    from land_grab import cli

    calls = []
    real = cli.plan_loop
    monkeypatch.setattr(cli, "plan_loop", lambda *a, **k: calls.append(k) or real(*a, **k))
    monkeypatch.setattr(pipeline, "N_CANDIDATES", 20)
    assert cli.main(["--postcode", "SW1A 1AA", "--km", "4"]) == 0
    assert calls and not calls[0].get("planner_only")


# --- cells you already own ---------------------------------------------------------------------


def test_claimed_cells_are_not_counted_and_are_kept_for_the_map(stubbed):
    plain = plan_loop("SW1A 1AA", 4, "walk", planner_only=True)
    owned = sorted(plain.safe_hexes)[:2]
    loop = plan_loop("SW1A 1AA", 4, "walk", planner_only=True, claimed=owned)

    assert loop.claimed == frozenset(owned)
    assert not (loop.safe_hexes & loop.claimed)
    assert not (loop.risky_hexes & loop.claimed)
    assert "leaflet" in map_html(loop).lower()
    assert "already yours" in map_html(loop)


def test_claimed_cells_can_be_typed_as_text(stubbed):
    cell = neighbours(HOME_CELL)[0]
    loop = plan_loop("SW1A 1AA", 4, "walk", planner_only=True, claimed=f" {cell.upper()} , {HOME_CELL},")
    assert loop.claimed == frozenset({cell, HOME_CELL})


def test_nothing_is_claimed_by_default(stubbed):
    assert plan_loop("SW1A 1AA", 4, "walk", planner_only=True).claimed == frozenset()


def test_a_bad_cell_id_is_rejected_before_any_download(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("should not look anything up")

    monkeypatch.setattr(pipeline, "postcode_to_latlon", fail)
    monkeypatch.setattr(pipeline, "get_network", fail)
    with pytest.raises(InvalidCellError, match="not-a-cell"):
        plan_loop("SW1A 1AA", 4, claimed="not-a-cell")
