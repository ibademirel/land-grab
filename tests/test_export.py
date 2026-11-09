import gpxpy
import pytest

from land_grab.export import save_gpx

ROUTE = [(51.50101, -0.14157), (51.50201, -0.14057), (51.50301, -0.13957), (51.50101, -0.14157)]


def _load(path):
    with open(path, encoding="utf-8") as f:
        return gpxpy.parse(f)


def test_saved_file_reopens_with_the_same_number_of_points(tmp_path):
    out = tmp_path / "route.gpx"
    assert save_gpx(ROUTE, out, "Test loop") == out

    gpx = _load(out)
    assert sum(len(seg.points) for track in gpx.tracks for seg in track.segments) == len(ROUTE)


def test_one_track_with_one_segment_and_the_given_name(tmp_path):
    save_gpx(ROUTE, tmp_path / "route.gpx", "Test loop")
    gpx = _load(tmp_path / "route.gpx")

    assert len(gpx.tracks) == 1
    assert gpx.tracks[0].name == "Test loop"
    assert len(gpx.tracks[0].segments) == 1


def test_points_keep_their_order_and_coordinates(tmp_path):
    save_gpx(ROUTE, tmp_path / "route.gpx", "x")
    points = _load(tmp_path / "route.gpx").tracks[0].segments[0].points

    got = [value for p in points for value in (p.latitude, p.longitude)]
    assert got == pytest.approx([value for latlon in ROUTE for value in latlon])


def test_creates_missing_parent_folders(tmp_path):
    out = tmp_path / "a" / "b" / "route.gpx"
    save_gpx(ROUTE, out, "x")
    assert out.exists()


def test_accepts_a_str_path_and_overwrites(tmp_path):
    out = str(tmp_path / "route.gpx")
    save_gpx(ROUTE, out, "first")
    save_gpx(ROUTE[:2], out, "second")

    gpx = _load(out)
    assert gpx.tracks[0].name == "second"
    assert len(gpx.tracks[0].segments[0].points) == 2


def test_empty_route_still_makes_a_valid_file(tmp_path):
    save_gpx([], tmp_path / "route.gpx", "empty")
    gpx = _load(tmp_path / "route.gpx")
    assert len(gpx.tracks) == 1 and gpx.tracks[0].segments[0].points == []


def test_name_with_special_characters_is_escaped(tmp_path):
    save_gpx(ROUTE, tmp_path / "route.gpx", "Loop <10 km> & more")
    assert _load(tmp_path / "route.gpx").tracks[0].name == "Loop <10 km> & more"
