import requests

POSTCODES_IO_URL = "https://api.postcodes.io/postcodes/{postcode}"
REQUEST_TIMEOUT = 10


def postcode_to_latlon(postcode: str) -> tuple[float, float]:
    """Return the (latitude, longitude) of a UK postcode using postcodes.io.

    Spaces are stripped and the postcode is uppercased before lookup.
    Raises ValueError if the postcode is empty or not found.
    """
    cleaned = "".join(postcode.split()).upper()
    if not cleaned:
        raise ValueError("Postcode must not be empty")

    response = requests.get(
        POSTCODES_IO_URL.format(postcode=cleaned), timeout=REQUEST_TIMEOUT
    )
    if response.status_code == 404:
        raise ValueError(f"Postcode not found: {postcode!r}")
    response.raise_for_status()

    result = response.json().get("result") or {}
    latitude = result.get("latitude")
    longitude = result.get("longitude")
    if latitude is None or longitude is None:
        raise ValueError(f"No coordinates available for postcode: {postcode!r}")
    return float(latitude), float(longitude)
