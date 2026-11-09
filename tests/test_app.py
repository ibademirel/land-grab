"""Tests for app.py using Streamlit's own test runner. Skipped where streamlit is not installed."""

import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from land_grab import pipeline, planner, web  # noqa: E402
from land_grab.hexgrid import _TO_WGS84, neighbours  # noqa: E402
from land_grab.router import LoopNotFoundError  # noqa: E402
from test_optimiser import _E0, _N0, HOME_CELL, _grid  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")

_LON, _LAT = _TO_WGS84.transform(_E0 - 1800, _N0 - 900)


@pytest.fixture(autouse=True)
def fresh_cache():
    web.ROUTES.clear()
    yield
    web.ROUTES.clear()


@pytest.fixture
def stubbed(monkeypatch):
    graph = _grid()
    monkeypatch.setattr(pipeline, "postcode_to_latlon", lambda postcode: (_LAT, _LON))
    monkeypatch.setattr(pipeline, "get_network", lambda *a, **k: graph)
    monkeypatch.setattr(pipeline.ox.distance, "nearest_nodes", lambda graph, X, Y: (2, 1))


def _app():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def _hexagons(at):
    return int(at.metric[0].value.split()[0])


def _plan(at, postcode="SW1A 1AA", km=4.0, activity="run", claimed=""):
    at.text_input[0].set_value(postcode)
    at.text_area[0].set_value(claimed)
    at.number_input[0].set_value(km)
    at.button_group[0].set_value(activity)
    at.button[0].click().run()
    return at


def test_the_page_has_the_title_and_one_tidy_input_block():
    at = _app()
    assert not at.exception
    assert "Land Grab Route Planner" in at.markdown[1].value
    assert at.text_input[0].label == "Postcode"
    assert at.number_input[0].min == 1.0
    assert list(at.button_group[0].options) == ["Run", "Walk", "Ride"]  # shown with an emoji each
    assert at.button_group[0].value == "run"
    assert at.button[0].label == "Plan my route"
    assert at.text_area[0].label == "Cells already claimed (yours or rivals'), optional" and at.text_area[0].value == ""
    assert not at.metric and not at.error


def test_a_route_shows_two_cards_and_a_collapsed_details_section(stubbed):
    at = _plan(_app(), km=4.0, activity="walk")
    assert not at.exception and not at.error

    assert [m.label for m in at.metric] == ["Hexagons", "Distance"]
    hexagons, points = (int(n) for n in re.fullmatch(r"(\d+) hexagons = (\d+) points", at.metric[0].value).groups())
    assert hexagons >= 1 and points == hexagons  # one point per hexagon (not the number of GPS points)
    assert at.metric[1].value.endswith(" km")

    details = [e for e in at.expander if e.label == "Details"]
    assert len(details) == 1
    assert details[0].proto.expanded is False
    assert "random search" in details[0].code[0].value and "skipped" in details[0].code[0].value


def test_a_good_route_shows_no_warnings(stubbed):
    at = _plan(_app())
    assert not at.warning


def test_problems_with_the_route_are_shown_as_warnings(stubbed, monkeypatch):
    from land_grab.checks import Problem

    monkeypatch.setattr(
        pipeline.PlannedLoop, "problems", property(lambda self: [Problem("gps_jump", "Jump!"), Problem("too_short", "Short!")])
    )
    at = _plan(_app())
    assert not at.exception and not at.error
    assert [w.value for w in at.warning] == ["Jump!", "Short!"]
    assert at.metric  # the route is still shown


def test_the_progress_steps_are_shown_in_words(stubbed):
    at = _plan(_app())
    steps = " ".join(str(w.value) for w in at.status[0].markdown)
    for step in ("Finding your postcode", "Loading the streets", "Choosing hexagons", "Joining them up", "Drawing your map"):
        assert step in steps
    assert at.status[0].label == "Your route is ready"


def test_asking_again_is_instant_and_says_so(stubbed, monkeypatch):
    at = _plan(_app())
    monkeypatch.setattr(pipeline, "plan_loop", lambda *a, **k: pytest.fail("must come from the cache"))
    at.button[0].click().run()
    assert not at.exception and not at.error
    assert "earlier search" in at.status[0].label
    assert [m.label for m in at.metric] == ["Hexagons", "Distance"]


def test_the_web_app_never_runs_the_random_search(stubbed, monkeypatch):
    monkeypatch.setattr(planner, "optimise_route", lambda *a, **k: pytest.fail("random search must not run"))
    at = _plan(_app())
    assert not at.exception and at.metric


def test_invalid_postcode_shows_a_friendly_message(monkeypatch):
    def not_found(postcode):
        raise ValueError("Postcode not found")

    monkeypatch.setattr(pipeline, "postcode_to_latlon", not_found)
    at = _plan(_app(), postcode="ZZ99 9ZZ")

    assert not at.exception
    assert len(at.error) == 1 and "couldn't find the postcode" in at.error[0].value
    assert not at.metric


def test_no_route_shows_a_friendly_message(monkeypatch):
    def none(*args, **kwargs):
        raise LoopNotFoundError("nothing")

    monkeypatch.setattr(pipeline, "plan_loop", none)
    at = _plan(_app())
    assert "couldn't find a route" in at.error[0].value


def test_a_surprise_failure_is_friendly_and_the_detail_is_inside_details(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret detail")

    monkeypatch.setattr(pipeline, "plan_loop", boom)
    at = _plan(_app())
    assert not at.exception
    assert "Something went wrong" in at.error[0].value and "secret detail" not in at.error[0].value
    assert "secret detail" in [e for e in at.expander if e.label == "Details"][0].code[0].value


# --- cells already claimed ---------------------------------------------------------------


def test_pasted_cells_are_shown_on_the_map_and_not_counted(stubbed):
    plain = _plan(_app())
    owned = sorted(neighbours(HOME_CELL))[:2]
    at = _plan(_app(), claimed=", ".join(owned))

    assert not at.exception and not at.error
    assert _hexagons(at) <= _hexagons(plain) + 1
    details = [e for e in at.expander if e.label == "Details"][0].code[0].value
    assert "cells already claimed (not counted as new): 2" in details


def test_a_mistyped_cell_id_shows_a_friendly_message(stubbed):
    at = _plan(_app(), claimed="88194ad147fffff, oops")

    assert not at.exception
    assert len(at.error) == 1 and '"oops"' in at.error[0].value
    assert not at.metric


# --- the theme ---------------------------------------------------------------------------


def test_the_theme_hides_the_menu_and_sets_calm_colours():
    import tomllib

    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert config["client"]["toolbarMode"] == "minimal"
    assert config["theme"]["primaryColor"].startswith("#") and config["theme"]["backgroundColor"].startswith("#")


def test_the_css_hides_the_footer_and_handles_phones():
    source = (ROOT / "app.py").read_text()
    assert "footer" in source and "#MainMenu" in source
    assert "@media (max-width: 640px)" in source


# --- ready for the web -------------------------------------------------------------------


def test_a_route_that_is_too_far_for_the_web_gives_a_friendly_message(stubbed):
    web_limit = web.MAX_KM["walk"]
    at = _app()
    at.number_input[0].set_value(max(web.MAX_KM.values()))  # the box itself stops at the biggest limit
    at.button_group[0].set_value("walk")
    at.button[0].click().run()
    assert not at.exception and len(at.error) == 1
    assert f"{web_limit:g} km" in at.error[0].value and "command line" in at.error[0].value
    assert not at.metric


def test_the_distance_box_stops_at_the_biggest_web_limit():
    at = _app()
    assert at.number_input[0].max == max(web.MAX_KM.values())


def test_the_map_is_embedded_with_st_iframe_not_the_removed_components_html():
    source = (ROOT / "app.py").read_text()
    assert "st.iframe(" in source
    assert "components.html" not in source and "streamlit.components" not in source


def test_the_page_has_no_fixed_paths_from_this_computer():
    for path in [ROOT / "app.py", *(ROOT / "land_grab").glob("*.py")]:
        text = path.read_text()
        assert "/Users/" not in text and "C:\\" not in text, path.name


def test_requirements_pin_everything_the_app_imports():
    lines = [l.split("#")[0].strip() for l in (ROOT / "requirements.txt").read_text().splitlines()]
    pinned = {l.split("==")[0].lower() for l in lines if l}
    assert all("==" in l for l in lines if l), "every line should be an exact version"
    for needed in ("streamlit", "h3", "osmnx", "scikit-learn", "networkx", "folium", "gpxpy", "pyproj", "shapely", "requests"):
        assert needed in pinned


def test_gitignore_hides_local_files_but_keeps_the_app():
    patterns = [l.strip() for l in (ROOT / ".gitignore").read_text().splitlines() if l.strip() and not l.startswith("#")]
    for hidden in (".venv/", "cache/", "/my_route.*", "/demo_*"):
        assert hidden in patterns
    for kept in ("app.py", "land_grab", "tests", "requirements.txt", ".streamlit", "config.toml"):
        assert kept not in (p.strip("/") for p in patterns), kept  # (".streamlit/secrets.toml" alone is fine)
    assert ".streamlit/config.toml" not in patterns and ".streamlit/*" not in patterns


# --- a busy map server -------------------------------------------------------------------


def test_a_busy_map_server_shows_the_busy_message_and_no_traceback(monkeypatch):
    from osmnx._errors import ResponseStatusCodeError

    def busy(*args, **kwargs):
        raise ResponseStatusCodeError("'overpass-api.de' responded: 502 Bad Gateway ... at capacity")

    monkeypatch.setattr(pipeline, "postcode_to_latlon", lambda postcode: (_LAT, _LON))
    monkeypatch.setattr(pipeline, "get_network", busy)
    at = _plan(_app(), km=5.0)

    assert not at.exception and len(at.error) == 1
    assert at.error[0].value == "The map service is busy right now. Please try again in a minute."
    assert not at.metric


def test_the_busy_retry_message_is_shown_while_waiting_and_the_route_still_arrives(stubbed, monkeypatch):
    graph = _grid()

    def slow(lat, lon, activity, radius_m, on_busy=None, **kwargs):
        on_busy()
        return graph

    monkeypatch.setattr(pipeline, "get_network", slow)
    at = _plan(_app())
    assert not at.exception and not at.error and at.metric
    steps = " ".join(str(w.value) for w in at.status[0].markdown)
    assert "Map service is busy, trying again..." in steps
