"""Sanity checks on a finished route, before it is sent to the challenge.

``validate_route`` returns a list of problems (empty if the route looks fine).
The messages are written for people, so the CLI and the web app can show them as they are.
"""

import math
from dataclasses import dataclass

LatLon = tuple[float, float]

MIN_ROUTE_KM = 1.0
LENGTH_TOLERANCE = 0.10  # the length may be this far from the requested km
MAX_JUMP_M = 200.0  # a bigger gap between two points is a "GPS jump"
CLOSED_WITHIN_M = 1.0  # start and end closer than this count as the same point
SPEED_RANGES_KMH = {"run": (6.0, 15.0), "walk": (3.0, 7.0), "ride": (12.0, 40.0)}

_EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class Problem:
    """One thing wrong with a route. ``rule`` is a short stable name; ``message`` is for people."""

    rule: str  # "too_short", "not_closed", "wrong_length", "gps_jump" or "bad_speed"
    message: str

    def __str__(self) -> str:
        return self.message


def distance_m(a: LatLon, b: LatLon) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    half_dlat = (lat2 - lat1) / 2
    half_dlon = math.radians(b[1] - a[1]) / 2
    h = math.sin(half_dlat) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(half_dlon) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(h))


def route_length_m(route: list[LatLon]) -> float:
    """Length of the route in metres, point to point."""
    return sum(distance_m(a, b) for a, b in zip(route, route[1:]))


def validate_route(
    route: list[LatLon],
    km: float,
    activity: str,
    *,
    duration_s: float | None = None,
) -> list[Problem]:
    """Check a route of (lat, lon) points against the challenge's rules; return the problems found.

    - ``too_short``: shorter than 1 km.
    - ``not_closed``: the end is not the start (within 1 m).
    - ``wrong_length``: more than 10% away from the requested ``km``.
    - ``gps_jump``: two points in a row more than 200 m apart.
    - ``bad_speed``: the average speed is outside the range for ``activity``
      (run 6-15 km/h, walk 3-7, ride 12-40). A route has no times of its own, so
      this needs ``duration_s``, the time the route took; without it, this check is skipped.

    Raises ValueError for a ``km`` that is not positive, an unknown ``activity``,
    or a ``duration_s`` that is not positive.
    """
    if not km > 0:
        raise ValueError(f"km must be positive, got {km}")
    if activity not in SPEED_RANGES_KMH:
        raise ValueError(f"activity must be one of {', '.join(SPEED_RANGES_KMH)}, got {activity!r}")
    if duration_s is not None and not duration_s > 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")

    if len(route) == 0:
        return [Problem("too_short", "The route is empty, so there is nothing to check.")]

    problems: list[Problem] = []
    length_m = route_length_m(route)

    if length_m < MIN_ROUTE_KM * 1000:
        problems.append(
            Problem(
                "too_short",
                f"This route is only {length_m / 1000:.2f} km long. "
                f"The challenge needs at least {MIN_ROUTE_KM:g} km.",
            )
        )

    gap = distance_m(route[0], route[-1])
    if gap > CLOSED_WITHIN_M:
        problems.append(
            Problem(
                "not_closed",
                f"The route doesn't finish where it started: the end is {gap:.0f} m from the start.",
            )
        )

    off = abs(length_m - km * 1000) / (km * 1000)
    if off > LENGTH_TOLERANCE:
        problems.append(
            Problem(
                "wrong_length",
                f"The route is {length_m / 1000:.2f} km, which is {off:.0%} away from the {km:g} km you asked for "
                f"(up to {LENGTH_TOLERANCE:.0%} is fine).",
            )
        )

    jumps = [
        (distance_m(a, b), i) for i, (a, b) in enumerate(zip(route, route[1:])) if distance_m(a, b) > MAX_JUMP_M
    ]
    if jumps:
        biggest, at = max(jumps)
        lat, lon = route[at]
        problems.append(
            Problem(
                "gps_jump",
                f"The route has {len(jumps)} sudden jump{'s' if len(jumps) != 1 else ''} of more than "
                f"{MAX_JUMP_M:g} m between points (the biggest is {biggest:.0f} m, near {lat:.5f}, {lon:.5f}). "
                "The challenge treats these as GPS jumps.",
            )
        )

    if duration_s is not None:
        speed = length_m / 1000 / (duration_s / 3600)
        low, high = SPEED_RANGES_KMH[activity]
        if not low <= speed <= high:
            problems.append(
                Problem(
                    "bad_speed",
                    f"An average speed of {speed:.1f} km/h doesn't look right for a {activity} "
                    f"(expected {low:g} to {high:g} km/h).",
                )
            )

    return problems
