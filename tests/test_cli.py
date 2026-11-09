import gpxpy
import pytest

from land_grab.router import LoopNotFoundError

from land_grab import cli, pipeline
from land_grab.cli import MAX_RADIUS_M, build_parser, main, network_radius_m, uncapped_network_radius_m
from land_grab.hexgrid import _TO_WGS84, neighbours
from test_optimiser import _E0, _N0, HOME_CELL, _grid

# --- network_radius_m ------------------------------------------------------------


def test_radius_is_half_the_length_plus_ten_percent():
    assert network_radius_m(10) == pytest.approx(5500)
    assert network_radius_m(1) == pytest.approx(550)
    assert network_radius_m(2.5) == pytest.approx(1375)


def test_radius_is_capped_at_15_km():
    assert network_radius_m(40) == MAX_RADIUS_M == 15_000
    assert network_radius_m(1000) == 15_000


def test_radius_is_not_capped_just_below_the_cap():
    assert network_radius_m(27) == pytest.approx(14_850)
    assert network_radius_m(27.5) == 15_000  # 15,125 m uncapped


def test_uncapped_radius_shows_when_the_cap_was_used():
    assert uncapped_network_radius_m(40) == pytest.approx(22_000)
    assert uncapped_network_radius_m(10) == network_radius_m(10)


@pytest.mark.parametrize("km", [0, -3, float("nan")])
def test_radius_rejects_non_positive_length(km):
    with pytest.raises(ValueError):
        network_radius_m(km)


# --- argument parsing --------------------------------------------------------------

BASE = ["--postcode", "SW1A 1AA", "--km", "10"]


def test_defaults():
    args = build_parser().parse_args(BASE)
    assert (args.postcode, args.km) == ("SW1A 1AA", 10.0)
    assert (args.activity, args.hex_size, args.seed) == ("run", None, 0)
    assert args.claimed == ""
    assert args.html is None and args.gpx is None


def test_all_options():
    args = build_parser().parse_args(
        BASE + ["--activity", "ride", "--hex-size", "600", "--seed", "7", "--html", "m.html", "--gpx", "r.gpx",
         "--claimed", "88194ad147fffff,88194ad141fffff"]
    )
    assert (args.activity, args.hex_size, args.seed) == ("ride", 600, 7)
    assert args.claimed == "88194ad147fffff,88194ad141fffff"
    assert (args.html, args.gpx) == ("m.html", "r.gpx")


@pytest.mark.parametrize(
    "argv",
    [
        ["--km", "10"],  # no postcode
        ["--postcode", "SW1A 1AA"],  # no km
        BASE[:3] + ["0.5"],  # km below 1
        BASE[:3] + ["-4"],
        BASE[:3] + ["abc"],
        BASE[:3] + ["nan"],
        BASE + ["--activity", "swim"],
        BASE + ["--hex-size", "0"],
    ],
)
def test_bad_arguments_exit_with_an_error(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(argv)
    assert exc.value.code == 2


def test_km_of_exactly_one_is_allowed():
    assert build_parser().parse_args(BASE[:3] + ["1"]).km == 1.0


# --- the whole command, with the postcode lookup and download stubbed -----------------

_LON, _LAT = _TO_WGS84.transform(_E0 - 1800, _N0 - 900)  # grid node (2, 1)


@pytest.fixture
def stubbed(monkeypatch):
    calls = {}
    graph = _grid()

    def fake_get_network(lat, lon, activity, radius_m, *a, **k):
        calls["network"] = (activity, radius_m)
        return graph

    monkeypatch.setattr(pipeline, "postcode_to_latlon", lambda postcode: (_LAT, _LON))
    monkeypatch.setattr(pipeline, "get_network", fake_get_network)
    monkeypatch.setattr(pipeline, "N_CANDIDATES", 20)
    # osmnx's nearest_nodes needs a real projected graph; the synthetic grid's start is node (2, 1).
    monkeypatch.setattr(pipeline.ox.distance, "nearest_nodes", lambda graph, X, Y: (2, 1))
    return calls


def test_command_prints_the_summary_and_saves_both_files(stubbed, tmp_path, capsys):
    html, gpx = tmp_path / "out" / "map.html", tmp_path / "out" / "route.gpx"
    code = main(["--postcode", "SW1A 1AA", "--km", "4", "--activity", "walk", "--html", str(html), "--gpx", str(gpx)])

    out = capsys.readouterr().out
    assert code == 0
    assert f"start: {_LAT:.5f}, {_LON:.5f}" in out
    assert "route length: " in out and " km" in out
    assert "new hexagons covered: " in out
    assert f"saved: {html}" in out and f"saved: {gpx}" in out
    assert "capped" not in out
    assert html.exists() and gpx.exists()
    assert stubbed["network"] == ("walk", pytest.approx(2200))
    with open(gpx) as f:
        assert len(gpxpy.parse(f).tracks[0].segments[0].points) > 2


def test_command_without_output_options_saves_nothing(stubbed, capsys):
    assert main(["--postcode", "SW1A 1AA", "--km", "4"]) == 0
    out = capsys.readouterr().out
    assert "new hexagons covered: " in out
    assert "nothing saved" in out and "saved:" not in out.replace("nothing saved", "")


def test_note_is_printed_when_the_radius_cap_is_used(stubbed, capsys, monkeypatch):
    # The synthetic grid is tiny, so stop after the download step.
    monkeypatch.setattr(pipeline, "choose_route", lambda *a, **k: (_ for _ in ()).throw(LoopNotFoundError("stop")))
    assert main(["--postcode", "SW1A 1AA", "--km", "40", "--activity", "ride"]) == 1
    captured = capsys.readouterr()
    assert "capped at 15 km" in captured.out
    assert stubbed["network"] == ("ride", 15_000)


def test_unknown_postcode_gives_a_clean_error(monkeypatch, capsys):
    def not_found(postcode):
        raise ValueError(f"Postcode not found: {postcode!r}")

    monkeypatch.setattr(pipeline, "postcode_to_latlon", not_found)
    assert main(["--postcode", "ZZ99 9ZZ", "--km", "5"]) == 1
    assert "error: Postcode not found" in capsys.readouterr().err


def test_no_loop_found_gives_a_clean_error(stubbed, capsys):
    assert main(["--postcode", "SW1A 1AA", "--km", "30"]) == 1  # far too long for the small grid
    assert "error:" in capsys.readouterr().err


def test_claimed_cells_are_not_counted_and_are_reported(stubbed, tmp_path, capsys):
    owned = sorted(neighbours(HOME_CELL))[:2]
    code = main(["--postcode", "SW1A 1AA", "--km", "4", "--claimed", ", ".join(owned) + ",", "--html", str(tmp_path / "m.html")])

    out = capsys.readouterr().out
    assert code == 0
    assert "claimed cells (already yours, not counted): 2" in out
    assert "already yours" in (tmp_path / "m.html").read_text()  # shown on the map in its own colour


def test_without_claimed_there_is_no_claimed_line(stubbed, capsys):
    assert main(["--postcode", "SW1A 1AA", "--km", "4"]) == 0
    assert "claimed cells" not in capsys.readouterr().out


def test_a_mistyped_cell_id_is_a_clean_error_before_anything_is_downloaded(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise AssertionError("should not download anything")

    monkeypatch.setattr(pipeline, "postcode_to_latlon", fail)
    monkeypatch.setattr(pipeline, "get_network", fail)
    assert main(["--postcode", "SW1A 1AA", "--km", "4", "--claimed", "88194ad147fffff,oops"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and '"oops"' in err and "88194ad147fffff" in err.split("oops")[1]


def test_hex_size_is_ignored_with_a_note(stubbed, capsys):
    assert main(["--postcode", "SW1A 1AA", "--km", "4", "--hex-size", "600"]) == 0
    assert "--hex-size is ignored" in capsys.readouterr().err


def test_python_dash_m_entry_point_exists():
    import importlib.util

    assert importlib.util.find_spec("land_grab.__main__") is not None


# --- live: real postcodes.io and OpenStreetMap, needs internet ---------------------------


@pytest.mark.live
def test_live_end_to_end(tmp_path, capsys):
    html, gpx = tmp_path / "map.html", tmp_path / "route.gpx"
    code = main(
        ["--postcode", "SW1A 1AA", "--km", "3", "--activity", "run", "--html", str(html), "--gpx", str(gpx)]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "start: 51.50" in out
    assert "route length: " in out
    hexagons = int(out.split("new hexagons covered: ")[1].split()[0])
    assert hexagons >= 1
    assert html.stat().st_size > 0
    with open(gpx) as f:
        points = gpxpy.parse(f).tracks[0].segments[0].points
    assert len(points) > 10
    assert (points[0].latitude, points[0].longitude) == (points[-1].latitude, points[-1].longitude)


# --- route checks -------------------------------------------------------------------


def test_cli_prints_no_warning_for_a_good_route(monkeypatch, capsys):
    _stub_pipeline(monkeypatch)
    assert main(["--postcode", "SW1A 1AA", "--km", "4"]) == 0
    assert "warning" not in capsys.readouterr().err


def test_cli_prints_each_problem_as_a_warning(monkeypatch, capsys):
    from land_grab.checks import Problem

    _stub_pipeline(monkeypatch)
    monkeypatch.setattr(
        pipeline.PlannedLoop, "problems", property(lambda self: [Problem("gps_jump", "Jump!"), Problem("too_short", "Short!")])
    )
    assert main(["--postcode", "SW1A 1AA", "--km", "4"]) == 0  # a warning does not fail the run
    err = capsys.readouterr().err
    assert "warning: Jump!" in err and "warning: Short!" in err


def test_cli_warns_when_the_route_is_the_wrong_length(monkeypatch, capsys):
    _stub_pipeline(monkeypatch)
    # The route comes out at about 4 km; have the checker believe 10 km was asked for.
    real = pipeline.validate_route
    monkeypatch.setattr(pipeline, "validate_route", lambda route, km, activity: real(route, 10.0, activity))
    main(["--postcode", "SW1A 1AA", "--km", "4"])
    assert "warning: The route is" in capsys.readouterr().err


def _stub_pipeline(monkeypatch):
    lon, lat = _TO_WGS84.transform(_E0 - 1800, _N0 - 900)
    graph = _grid()
    monkeypatch.setattr(pipeline, "postcode_to_latlon", lambda postcode: (lat, lon))
    monkeypatch.setattr(pipeline, "get_network", lambda *a, **k: graph)
    monkeypatch.setattr(pipeline.ox.distance, "nearest_nodes", lambda graph, X, Y: (2, 1))
    monkeypatch.setattr(pipeline, "N_CANDIDATES", 20)
