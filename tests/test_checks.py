import pytest

from land_grab.checks import (
    MAX_JUMP_M,
    SPEED_RANGES_KMH,
    Problem,
    distance_m,
    route_length_m,
    validate_route,
)

M_PER_DEG_LAT = 111_195.0  # about; the tests only need it to be consistent


def _out_and_back(km, step_m=50.0, lon=-0.14):
    """A closed route: north for km/2, then back the same way, a point every ``step_m``."""
    half = km * 1000 / 2
    n = int(half // step_m)
    out = [(51.5 + i * step_m / M_PER_DEG_LAT, lon) for i in range(n + 1)]
    end = (51.5 + half / M_PER_DEG_LAT, lon)
    return out + [end] + out[::-1]


def _rules(problems):
    return [p.rule for p in problems]


@pytest.fixture
def good():
    return _out_and_back(5.0)


# --- helpers -------------------------------------------------------------------------


def test_distance_and_length():
    assert distance_m((51.5, -0.1), (51.5, -0.1)) == 0
    assert distance_m((51.0, 0.0), (52.0, 0.0)) == pytest.approx(111_195, rel=1e-3)
    assert route_length_m(_out_and_back(5.0)) == pytest.approx(5000, rel=0.01)
    assert route_length_m([]) == 0 and route_length_m([(51.5, 0.0)]) == 0


# --- a good route ----------------------------------------------------------------------


@pytest.mark.parametrize("activity", ["run", "walk", "ride"])
def test_a_good_route_has_no_problems(good, activity):
    assert validate_route(good, 5.0, activity) == []


def test_problems_print_as_plain_sentences(good):
    problem = validate_route(good[:-1], 5.0, "run")[0]
    assert isinstance(problem, Problem) and str(problem) == problem.message
    assert problem.message[0].isupper() and problem.message.endswith(".")


# --- rule 1: shorter than 1 km ---------------------------------------------------------------


def test_route_shorter_than_1_km_is_flagged():
    short = _out_and_back(0.8)
    assert "too_short" in _rules(validate_route(short, 0.8, "run"))


def test_route_of_exactly_1_km_is_not_too_short():
    assert "too_short" not in _rules(validate_route(_out_and_back(1.0), 1.0, "run"))


def test_an_empty_route_is_flagged_and_does_not_crash():
    assert _rules(validate_route([], 5.0, "run")) == ["too_short"]


def test_a_single_point_is_too_short_and_off_length():
    rules = _rules(validate_route([(51.5, 0.0)], 5.0, "run"))
    assert "too_short" in rules and "wrong_length" in rules and "not_closed" not in rules


# --- rule 2: not closed ----------------------------------------------------------------------


def test_route_that_does_not_end_at_the_start_is_flagged(good):
    problems = validate_route(good[:-60], 5.0, "run")
    assert "not_closed" in _rules(problems)
    assert "doesn't finish where it started" in next(p for p in problems if p.rule == "not_closed").message


def test_route_ending_within_a_metre_of_the_start_counts_as_closed(good):
    nearly = good[:-1] + [(good[0][0] + 0.5 / M_PER_DEG_LAT, good[0][1])]
    assert "not_closed" not in _rules(validate_route(nearly, 5.0, "run"))
    far = good[:-1] + [(good[0][0] + 5 / M_PER_DEG_LAT, good[0][1])]
    assert "not_closed" in _rules(validate_route(far, 5.0, "run"))


# --- rule 3: length more than 10% away -------------------------------------------------------------


def test_length_just_inside_and_just_outside_the_10_percent_band():
    route = _out_and_back(5.0)
    length_km = route_length_m(route) / 1000
    assert "wrong_length" not in _rules(validate_route(route, length_km / 0.91, "run"))  # 9% under the request
    assert "wrong_length" not in _rules(validate_route(route, length_km / 1.09, "run"))  # 9% over
    assert "wrong_length" in _rules(validate_route(route, length_km / 0.89, "run"))
    assert "wrong_length" in _rules(validate_route(route, length_km / 1.11, "run"))


def test_the_length_message_says_how_far_off_it_is(good):
    message = next(p for p in validate_route(good, 8.0, "run") if p.rule == "wrong_length").message
    assert "5.00 km" in message and "8 km" in message and "%" in message


# --- rule 4: GPS jumps ----------------------------------------------------------------------


def test_a_jump_of_more_than_200_m_is_flagged(good):
    # Drop 20 points from the outward leg: a gap of about 1 km.
    jumpy = good[:10] + good[30:]
    problems = validate_route(jumpy, 5.0, "run")
    jump = next(p for p in problems if p.rule == "gps_jump")
    assert "1 sudden jump" in jump.message and "GPS jump" in jump.message


def test_jumps_are_counted_and_the_biggest_is_reported():
    route = [(51.5, 0.0), (51.5 + 250 / M_PER_DEG_LAT, 0.0), (51.5 + 750 / M_PER_DEG_LAT, 0.0), (51.5, 0.0)]
    message = next(p for p in validate_route(route, 1.0, "run") if p.rule == "gps_jump").message
    assert "3 sudden jumps" in message and "biggest is 750 m" in message


def test_a_gap_just_under_200_m_is_fine_and_just_over_is_not():
    def route(gap_m):
        return [(51.5, 0.0), (51.5 + gap_m / M_PER_DEG_LAT, 0.0), (51.5, 0.0)]

    assert MAX_JUMP_M == 200.0
    assert "gps_jump" not in _rules(validate_route(route(199), 0.4, "run"))
    assert "gps_jump" in _rules(validate_route(route(202), 0.4, "run"))


def test_the_jump_check_looks_at_the_closing_leg_too():
    route = _out_and_back(5.0)[:-1] + [(51.5 + 0.0001, -0.14)]
    assert "gps_jump" not in _rules(validate_route(route, 5.0, "run"))  # tiny end gap: no jump


# --- rule 5: average speed ----------------------------------------------------------------


@pytest.mark.parametrize(
    "activity,fine_kmh,slow_kmh,fast_kmh",
    [("run", 10, 5, 16), ("walk", 5, 2, 8), ("ride", 25, 11, 41)],
)
def test_speed_is_checked_against_the_range_for_the_activity(activity, fine_kmh, slow_kmh, fast_kmh):
    route = _out_and_back(5.0)
    hours = lambda kmh: 5.0 / kmh * 3600  # noqa: E731
    assert validate_route(route, 5.0, activity, duration_s=hours(fine_kmh)) == []
    for kmh in (slow_kmh, fast_kmh):
        problems = validate_route(route, 5.0, activity, duration_s=hours(kmh))
        assert _rules(problems) == ["bad_speed"], kmh
        assert activity in problems[0].message


def test_speed_ranges_are_the_ones_the_challenge_uses():
    assert SPEED_RANGES_KMH == {"run": (6, 15), "walk": (3, 7), "ride": (12, 40)}


@pytest.mark.parametrize("activity", ["run", "walk", "ride"])
def test_the_edges_of_the_speed_range_are_allowed(activity):
    low, high = SPEED_RANGES_KMH[activity]
    route = _out_and_back(5.0)
    length_km = route_length_m(route) / 1000
    for kmh in (low, high):
        assert validate_route(route, 5.0, activity, duration_s=length_km / kmh * 3600) == []


def test_a_route_without_times_skips_the_speed_check(good):
    assert "bad_speed" not in _rules(validate_route(good, 5.0, "walk"))


# --- several problems at once, and bad arguments -----------------------------------------------


def test_several_problems_are_all_reported():
    route = [(51.5, 0.0), (51.5 + 500 / M_PER_DEG_LAT, 0.0)]
    assert _rules(validate_route(route, 5.0, "run")) == ["too_short", "not_closed", "wrong_length", "gps_jump"]


@pytest.mark.parametrize(
    "km,activity,duration",
    [(0, "run", None), (-2, "run", None), (float("nan"), "run", None), (5, "swim", None), (5, "run", 0), (5, "run", -1)],
)
def test_bad_arguments_raise_value_error(good, km, activity, duration):
    with pytest.raises(ValueError):
        validate_route(good, km, activity, duration_s=duration)
