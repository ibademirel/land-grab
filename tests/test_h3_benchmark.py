"""Checks against the REAL h3 package, using two cell IDs from the challenge map.

Skipped when the real package is not installed (tests/conftest.py then uses a stand-in, and a
stand-in cannot tell us anything about real H3). Run with ``-s`` to see the printed centres and
boundaries, so they can be compared with the challenge map:

    .venv/bin/python -m pytest tests/test_h3_benchmark.py -s
"""

import math

import pytest

import conftest

pytestmark = pytest.mark.skipif(
    conftest.USING_FAKE_H3, reason="the real h3 package is not installed (run: .venv/bin/python -m pip install 'h3>=4,<5')"
)

CELLS = ["88194ad147fffff", "88194ad141fffff"]


def _h3():
    import h3

    return h3


def _km_between(a, b):
    (lat1, lon1), (lat2, lon2) = a, b
    dx = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2))
    dy = math.radians(lat2 - lat1)
    return 6371 * math.hypot(dx, dy)


def test_print_the_two_cells_for_comparison_with_the_challenge_map(capsys):
    h3 = _h3()
    with capsys.disabled():
        print()
        for cell in CELLS:
            lat, lng = h3.cell_to_latlng(cell)
            print(f"cell {cell}: centre lat {lat:.6f}, lng {lng:.6f}")
            for i, (blat, blng) in enumerate(h3.cell_to_boundary(cell)):
                print(f"    corner {i}: lat {blat:.6f}, lng {blng:.6f}")


@pytest.mark.parametrize("cell", CELLS)
def test_cell_is_in_central_london_at_resolution_8(cell):
    h3 = _h3()
    lat, lng = h3.cell_to_latlng(cell)
    assert 51.4 <= lat <= 51.6
    assert -0.3 <= lng <= 0.1
    assert h3.get_resolution(cell) == 8
    assert h3.is_valid_cell(cell)


@pytest.mark.parametrize("cell", CELLS)
def test_the_centre_of_a_cell_gives_the_same_id_back(cell):
    h3 = _h3()
    lat, lng = h3.cell_to_latlng(cell)
    assert h3.latlng_to_cell(lat, lng, 8) == cell


def test_the_two_cells_are_close_together():
    h3 = _h3()
    a, b = (h3.cell_to_latlng(c) for c in CELLS)
    assert _km_between(a, b) < 5


def test_our_own_functions_agree_with_h3_for_these_cells():
    from land_grab.hexgrid import hex_boundary, hex_centre, is_valid_hex, latlon_to_hex, neighbours

    h3 = _h3()
    for cell in CELLS:
        assert is_valid_hex(cell)
        assert hex_centre(cell) == h3.cell_to_latlng(cell)
        assert hex_boundary(cell) == [tuple(p) for p in h3.cell_to_boundary(cell)]
        assert latlon_to_hex(*hex_centre(cell)) == cell
        assert len(neighbours(cell)) == 6
