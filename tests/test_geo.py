from unittest.mock import MagicMock, patch

import pytest
import requests

from land_grab.geo import postcode_to_latlon


def _response(status_code=200, payload=None):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload or {}
    if status_code >= 400:
        response.raise_for_status.side_effect = requests.HTTPError(
            f"{status_code} error"
        )
    return response


@patch("land_grab.geo.requests.get")
def test_returns_lat_lon(mock_get):
    mock_get.return_value = _response(
        payload={"status": 200, "result": {"latitude": 51.501009, "longitude": -0.141588}}
    )
    assert postcode_to_latlon("SW1A 1AA") == (51.501009, -0.141588)


@patch("land_grab.geo.requests.get")
def test_strips_spaces_and_uppercases(mock_get):
    mock_get.return_value = _response(
        payload={"result": {"latitude": 51.5, "longitude": -0.14}}
    )
    postcode_to_latlon("  sw1a 1aa ")
    url = mock_get.call_args.args[0]
    assert url == "https://api.postcodes.io/postcodes/SW1A1AA"


@patch("land_grab.geo.requests.get")
def test_uses_10_second_timeout(mock_get):
    mock_get.return_value = _response(
        payload={"result": {"latitude": 51.5, "longitude": -0.14}}
    )
    postcode_to_latlon("SW1A 1AA")
    assert mock_get.call_args.kwargs["timeout"] == 10


@patch("land_grab.geo.requests.get")
def test_not_found_raises_value_error(mock_get):
    mock_get.return_value = _response(404, {"status": 404, "error": "Invalid postcode"})
    with pytest.raises(ValueError, match="not found"):
        postcode_to_latlon("ZZ99 9ZZ")


@patch("land_grab.geo.requests.get")
def test_missing_coordinates_raises_value_error(mock_get):
    mock_get.return_value = _response(
        payload={"result": {"latitude": None, "longitude": None}}
    )
    with pytest.raises(ValueError, match="No coordinates"):
        postcode_to_latlon("AB1 2CD")


@patch("land_grab.geo.requests.get")
def test_server_error_propagates(mock_get):
    mock_get.return_value = _response(500)
    with pytest.raises(requests.HTTPError):
        postcode_to_latlon("SW1A 1AA")


@patch("land_grab.geo.requests.get")
def test_empty_postcode_raises_without_request(mock_get):
    with pytest.raises(ValueError, match="empty"):
        postcode_to_latlon("   ")
    mock_get.assert_not_called()


@pytest.mark.live
def test_live_sw1a_1aa():
    lat, lon = postcode_to_latlon("SW1A 1AA")
    assert lat == pytest.approx(51.50, abs=0.01)
    assert lon == pytest.approx(-0.14, abs=0.01)
