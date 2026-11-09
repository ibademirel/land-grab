"""Test set-up: if the real ``h3`` package is not installed, use the small stand-in in fake_h3.py.

On a machine with ``h3`` installed (the normal case; see requirements.txt) the
real package is used and the stand-in is never loaded.
"""

import sys

try:
    import h3  # noqa: F401

    USING_FAKE_H3 = False
except ImportError:
    import fake_h3

    sys.modules["h3"] = fake_h3
    USING_FAKE_H3 = True


def pytest_report_header(config):
    if USING_FAKE_H3:
        return "h3: NOT INSTALLED - using the stand-in in tests/fake_h3.py (the benchmark test is skipped)"
    return "h3: real package"


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def no_real_prebuilt_files(monkeypatch, tmp_path):
    """Tests never see the real ``prebuilt/`` folder or GitHub, so nothing on this computer or online changes their results."""
    from land_grab import network

    monkeypatch.setattr(network, "PREBUILT_DIR", tmp_path / "no-prebuilt-here")
    monkeypatch.setattr(network, "RELEASE_FILES", ())  # and never reach out to GitHub
    monkeypatch.setattr(network, "_release_failed_at", {})
