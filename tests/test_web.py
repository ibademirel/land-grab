"""The web app's logic (no Streamlit needed): the route cache, progress stages and friendly errors."""

import gpxpy
import pytest
import requests

from land_grab import pipeline, planner, web
from land_grab.hexgrid import _TO_WGS84, InvalidCellError, neighbours
from land_grab.pipeline import InvalidPostcodeError
from land_grab.router import LoopNotFoundError
from land_grab.web import MAX_KM, RouteCache, TooFarError, cache_key, details_text, friendly_error, get_route
from test_optimiser import _E0, _N0, HOME_CELL, _grid

_LON, _LAT = _TO_WGS84.transform(_E0 - 1800, _N0 - 900)


@pytest.fixture
def counted(monkeypatch):
    """The pipeline on the synthetic grid, counting lookups, downloads and planner runs."""
    graph = _grid()
    calls = {"postcode": 0, "network": 0, "planner": 0, "random": 0}

    def lookup(postcode):
        calls["postcode"] += 1
        return _LAT, _LON

    def network(*args, **kwargs):
        calls["network"] += 1
        return graph

    real_plan, real_random = planner.plan_route, planner.optimise_route
    monkeypatch.setattr(pipeline, "postcode_to_latlon", lookup)
    monkeypatch.setattr(pipeline, "get_network", network)
    monkeypatch.setattr(pipeline.ox.distance, "nearest_nodes", lambda graph, X, Y: (2, 1))
    monkeypatch.setattr(planner, "plan_route", lambda *a, **k: calls.__setitem__("planner", calls["planner"] + 1) or real_plan(*a, **k))
    monkeypatch.setattr(planner, "optimise_route", lambda *a, **k: calls.__setitem__("random", calls["random"] + 1) or real_random(*a, **k))
    return calls


@pytest.fixture
def cache():
    return RouteCache()


# --- the cache key ---------------------------------------------------------------------


def test_the_key_ignores_spaces_case_and_activity_case():
    assert cache_key("sw1a 1aa", 5, "Run") == cache_key(" SW1A1AA ", 5.0, "run") == ("SW1A1AA", 5.0, "run", frozenset())


def test_the_key_differs_by_postcode_km_and_activity():
    keys = {
        cache_key("SW1A 1AA", 5, "run"),
        cache_key("SW1A 1AB", 5, "run"),
        cache_key("SW1A 1AA", 6, "run"),
        cache_key("SW1A 1AA", 5, "ride"),
    }
    assert len(keys) == 4


# --- the cache itself ------------------------------------------------------------------


def test_cache_keeps_the_most_recently_used_routes():
    small = RouteCache(max_size=2)
    small.put(("A", 1.0, "run", frozenset()), "a")
    small.put(("B", 1.0, "run", frozenset()), "b")
    assert small.get(("A", 1.0, "run", frozenset())) == "a"  # A is now newer than B
    small.put(("C", 1.0, "run", frozenset()), "c")
    assert ("B", 1.0, "run", frozenset()) not in small and len(small) == 2
    assert small.get(("A", 1.0, "run", frozenset())) == "a" and small.get(("C", 1.0, "run", frozenset())) == "c"


def test_cache_clear_and_bad_size():
    small = RouteCache(max_size=2)
    small.put(("A", 1.0, "run", frozenset()), "a")
    small.clear()
    assert len(small) == 0 and small.get(("A", 1.0, "run", frozenset())) is None
    with pytest.raises(ValueError):
        RouteCache(max_size=0)


# --- get_route -------------------------------------------------------------------------


def test_the_second_request_for_the_same_route_is_instant(counted, cache):
    stages = []
    first, cached_first = get_route("SW1A 1AA", 4, "walk", on_stage=stages.append, cache=cache)
    assert not cached_first
    assert stages == [
        "Finding your postcode", "Loading the streets", "Choosing hexagons", "Joining them up", "Drawing your map",
    ]
    done = dict(counted)

    again_stages = []
    second, cached_second = get_route("SW1A 1AA", 4, "walk", on_stage=again_stages.append, cache=cache)
    assert cached_second and second is first
    assert counted == done  # no postcode lookup, no download, no planning
    assert again_stages == []


def test_the_same_route_written_differently_is_still_cached(counted, cache):
    get_route("SW1A 1AA", 4, "walk", cache=cache)
    done = dict(counted)
    _, cached = get_route("  sw1a1aa", 4.0, "Walk", cache=cache)
    assert cached and counted == done


@pytest.mark.parametrize("other", [("SW1A 1AB", 4, "walk"), ("SW1A 1AA", 5, "walk"), ("SW1A 1AA", 4, "run")])
def test_a_different_postcode_km_or_activity_is_planned_again(counted, cache, other):
    get_route("SW1A 1AA", 4, "walk", cache=cache)
    before = counted["planner"]
    _, cached = get_route(*other, cache=cache)
    assert not cached and counted["planner"] == before + 1 and len(cache) == 2


def test_failures_are_not_cached(monkeypatch, cache):
    def not_found(postcode):
        raise ValueError("Postcode not found")

    monkeypatch.setattr(pipeline, "postcode_to_latlon", not_found)
    for _ in range(2):
        with pytest.raises(InvalidPostcodeError):
            get_route("ZZ99 9ZZ", 5, "run", cache=cache)
    assert len(cache) == 0


def test_get_route_uses_the_shared_cache_by_default(counted):
    web.ROUTES.clear()
    try:
        get_route("SW1A 1AA", 4, "run")
        assert get_route("SW1A 1AA", 4, "run")[1] is True
    finally:
        web.ROUTES.clear()


# --- cells already claimed -----------------------------------------------------------------

_A, _B = sorted(neighbours(HOME_CELL))[:2]


def test_the_key_includes_the_claimed_cells_whatever_order_or_spelling():
    plain = cache_key("SW1A 1AA", 5, "run")
    one = cache_key("SW1A 1AA", 5, "run", f"{_A}, {_B}")
    assert one == cache_key("SW1A 1AA", 5, "run", f"{_B.upper()},{_A}") == cache_key("SW1A 1AA", 5, "run", [_B, _A])
    assert one[3] == frozenset({_A, _B}) and one != plain


def test_a_bad_cell_id_makes_the_key_fail_with_a_readable_message():
    with pytest.raises(InvalidCellError, match="nonsense"):
        cache_key("SW1A 1AA", 5, "run", "nonsense")


def test_different_claimed_cells_are_planned_again(counted, cache):
    get_route("SW1A 1AA", 4, "walk", cache=cache)
    before = counted["planner"]
    result, cached = get_route("SW1A 1AA", 4, "walk", _A, cache=cache)
    assert not cached and counted["planner"] == before + 1 and len(cache) == 2
    assert result.loop.claimed == frozenset({_A})
    assert get_route("SW1A 1AA", 4, "walk", _A, cache=cache)[1] is True


def test_claimed_cells_are_not_counted_in_the_hexagon_card_and_are_in_the_details(counted, cache):
    plain, _ = get_route("SW1A 1AA", 4, "walk", cache=cache)
    owned = ", ".join(sorted(plain.loop.safe_hexes)[:2])
    result, _ = get_route("SW1A 1AA", 4, "walk", owned, cache=cache)
    assert result.hexagons == len(result.loop.safe_hexes)
    assert not (result.loop.safe_hexes & result.loop.claimed)
    assert "cells already claimed (not counted as new): 2" in details_text(result)
    assert "cells already claimed" not in details_text(plain)


def test_a_bad_cell_id_has_a_friendly_message_that_names_it():
    message = friendly_error(InvalidCellError('These don\'t look like H3 cell IDs at resolution 8: "oops".'), "SW1A 1AA")
    assert '"oops"' in message and "Something went wrong" not in message


# --- planner only ----------------------------------------------------------------------


def test_the_web_app_runs_only_the_planner(counted, cache):
    result, _ = get_route("SW1A 1AA", 4, "run", cache=cache)
    assert counted["planner"] == 1 and counted["random"] == 0
    assert result.loop.choice.chosen is result.loop.choice.planner
    assert result.loop.choice.random.error == planner.SKIPPED


def test_if_the_planner_fails_a_short_route_uses_the_random_search(counted, cache, monkeypatch):
    monkeypatch.setattr(planner, "plan_route", lambda *a, **k: (_ for _ in ()).throw(planner.PlannerError("no plan")))
    result, _ = get_route("SW1A 1AA", 4, "run", cache=cache)
    assert counted["random"] == 1
    assert result.loop.choice.chosen is result.loop.choice.random
    assert "random search was used" in details_text(result)


def test_if_the_planner_fails_a_long_route_is_not_searched_randomly(counted, cache, monkeypatch):
    monkeypatch.setattr(planner, "plan_route", lambda *a, **k: (_ for _ in ()).throw(planner.PlannerError("no plan")))
    with pytest.raises(LoopNotFoundError):
        get_route("SW1A 1AA", 13, "ride", cache=cache)  # over 12 km is "long": no random search
    assert counted["random"] == 0 and len(cache) == 0


# --- what the screen shows ---------------------------------------------------------------


def test_the_result_has_the_three_numbers_the_map_and_the_gpx(counted, cache):
    result, _ = get_route("SW1A 1AA", 4, "walk", cache=cache)
    assert result.hexagons == len(result.loop.safe_hexes) >= 1
    assert result.distance_km == pytest.approx(4, rel=0.06)  # the planner may be up to 5% short or long
    assert result.points == len(result.loop.route_latlons) > 2
    assert "leaflet" in result.map_html.lower()
    assert len(gpxpy.parse(result.gpx).tracks[0].segments[0].points) == result.points


def test_details_text_has_the_table_and_where_it_came_from(counted, cache):
    result, _ = get_route("SW1A 1AA", 4, "walk", cache=cache)
    fresh = details_text(result)
    assert "planner" in fresh and "random search" in fresh and "skipped" in fresh
    assert "safe hexagons" in fresh and "H3 resolution 8" in fresh and "made in" in fresh
    assert "instantly" in details_text(result, came_from_cache=True)


# --- friendly errors -------------------------------------------------------------------


def test_every_error_has_a_plain_friendly_message():
    assert "couldn't find the postcode \"ZZ99\"" in friendly_error(InvalidPostcodeError("x"), " ZZ99 ")
    assert "type a postcode" in friendly_error(InvalidPostcodeError("x"), "  ")
    assert "couldn't find a route" in friendly_error(LoopNotFoundError("x"))
    assert "internet connection" in friendly_error(requests.ConnectionError("x"))
    assert "distance" in friendly_error(ValueError("km must be positive"))
    assert "Something went wrong" in friendly_error(RuntimeError("secret detail"))
    for error in (InvalidPostcodeError("x"), LoopNotFoundError("x"), requests.Timeout(), ValueError("v"), KeyError("k")):
        message = friendly_error(error, "SW1A")
        assert "Traceback" not in message and type(error).__name__ not in message


# --- the web-only distance limit ---------------------------------------------------------


@pytest.mark.parametrize("activity", ["run", "walk", "ride"])
def test_a_route_over_the_web_limit_is_refused_before_anything_is_looked_up(counted, cache, activity):
    with pytest.raises(TooFarError):
        get_route("SW1A 1AA", MAX_KM[activity] + 1, activity, cache=cache)
    assert counted["postcode"] == 0 and counted["network"] == 0 and len(cache) == 0


def test_the_limits_are_sensible_and_a_route_at_the_limit_is_allowed(counted, cache):
    assert MAX_KM["run"] <= 15 and MAX_KM["walk"] <= 15 and MAX_KM["ride"] <= 25
    web.check_distance(MAX_KM["ride"], "ride")  # exactly at the limit: no error
    web.check_distance(MAX_KM["run"], " RUN ")


def test_the_too_far_message_is_friendly_and_names_the_limit():
    with pytest.raises(TooFarError) as caught:
        web.check_distance(MAX_KM["run"] + 5, "run")
    message = friendly_error(caught.value, "SW1A 1AA")
    assert f"{MAX_KM['run']:g} km" in message and "run" in message and "command line" in message


def test_the_command_line_has_no_distance_limit():
    import inspect

    assert "MAX_KM" not in inspect.getsource(pipeline) and "check_distance" not in inspect.getsource(pipeline)


# --- a busy map server ---------------------------------------------------------------------

BUSY = "The map service is busy right now. Please try again in a minute."


def test_the_overpass_502_error_says_the_map_service_is_busy_not_that_the_distance_is_wrong():
    from osmnx._errors import ResponseStatusCodeError

    # The exact error the deployed app showed. It is a kind of ValueError, which friendly_error used to
    # read as "bad distance".
    assert issubclass(ResponseStatusCodeError, ValueError)
    error = ResponseStatusCodeError("'overpass-api.de' responded: 502 Bad Gateway The server is probably at capacity")
    assert friendly_error(error, "SW1A 1AA") == BUSY
    assert "distance" not in friendly_error(error, "SW1A 1AA")


def test_our_own_busy_error_and_an_empty_answer_have_friendly_messages():
    from osmnx._errors import InsufficientResponseError

    assert friendly_error(web.MapServiceBusyError(429, "https://overpass-api.de/api/interpreter")) == BUSY
    assert "didn't send back any streets" in friendly_error(InsufficientResponseError("empty"))
    assert "distance" in friendly_error(ValueError("km must be positive"))  # a genuine bad distance still says so


def test_a_busy_map_server_during_planning_is_shown_as_busy_and_is_not_cached(monkeypatch, cache):
    from osmnx._errors import ResponseStatusCodeError

    monkeypatch.setattr(pipeline, "postcode_to_latlon", lambda postcode: (_LAT, _LON))

    def busy_network(*args, **kwargs):
        raise ResponseStatusCodeError("'overpass-api.de' responded: 502 Bad Gateway")

    monkeypatch.setattr(pipeline, "get_network", busy_network)
    with pytest.raises(ResponseStatusCodeError) as caught:
        get_route("SW1A 1AA", 5, "run", cache=cache)
    assert friendly_error(caught.value, "SW1A 1AA") == BUSY
    assert len(cache) == 0


def test_the_busy_message_is_passed_on_as_a_progress_stage(monkeypatch, cache):
    monkeypatch.setattr(pipeline, "postcode_to_latlon", lambda postcode: (_LAT, _LON))
    graph = _grid()

    def slow_network(lat, lon, activity, radius_m, on_busy=None, **kwargs):
        on_busy()  # as if the first server was busy and we are about to wait
        return graph

    monkeypatch.setattr(pipeline, "get_network", slow_network)
    monkeypatch.setattr(pipeline.ox.distance, "nearest_nodes", lambda graph, X, Y: (2, 1))
    stages = []
    get_route("SW1A 1AA", 4, "walk", cache=cache, on_stage=stages.append)
    assert "Map service is busy, trying again..." in stages
    assert stages.index("Loading the streets") < stages.index("Map service is busy, trying again...")
