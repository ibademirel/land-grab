"""Save a route as a GPX file for a watch, phone or bike computer."""

from pathlib import Path

import gpxpy.gpx

LatLon = tuple[float, float]


def save_gpx(route_latlons: list[LatLon], out_path: str | Path, name: str) -> Path:
    """Save a route of (lat, lon) points as a GPX file with one track and one segment.

    The track is called ``name``. Missing parent folders of ``out_path`` are
    created. Returns the path written.
    """
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack(name=name)
    segment = gpxpy.gpx.GPXTrackSegment()
    segment.points.extend(
        gpxpy.gpx.GPXTrackPoint(latitude=lat, longitude=lon) for lat, lon in route_latlons
    )
    track.segments.append(segment)
    gpx.tracks.append(track)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(gpx.to_xml(), encoding="utf-8")
    return out_path
