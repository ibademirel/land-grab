"""Streets downloaded by the visitor's own phone or browser, instead of by our server.

The free cloud server is often refused by the public Overpass map servers ("map service busy"), but a
phone on its own connection usually is not. So when no ready-made file covers a place, the web page
asks the phone to fetch the streets and send them back. This file is the Python half:

* ``build_request`` writes the exact Overpass question the server would have asked (same area, same
  street filter, from OSMnx's own code), so the phone only has to send the text;
* ``graph_from_reply`` checks what the phone sent back and turns it into the same street network the
  server download makes (the same OSMnx steps: two-way walking streets, cut to the area, largest
  connected piece, simplified).

The phone's half is ``phone_streets.mjs`` (JavaScript). Kept free of Streamlit so it can be tested.
"""

import base64
import binascii
import gzip
import io
import json
import math
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx
import osmnx as ox
from osmnx import _overpass
from osmnx import settings as ox_settings

from land_grab.network import BUSY_STATUS_CODES, OVERPASS_SERVERS, RETRY_WAITS_S, street_filter

REPLY_FORMAT = 1  # bump this (here and in phone_streets.mjs) if the layout of the phone's reply changes
MAX_PAYLOAD_CHARS = 24_000_000  # Streamlit refuses messages from the browser over 25 MB, so the page stops at about this
MAX_JSON_BYTES = 400_000_000  # a safety stop for a reply that unpacks to something absurd
MAX_NODES = 3_000_000

PHONE_FAILED_MESSAGE = "Your phone couldn't download the street map. Trying our server instead…"
PHONE_PROGRESS_MESSAGE = "Downloading streets to your phone… keep this page open."


class PhoneStreetsError(Exception):
    """The phone could not fetch the streets, or what it sent back is not usable (the message is for the details box)."""


@dataclass(frozen=True)
class PhoneRequest:
    """Everything the phone needs: what to ask, and which servers to ask (in order)."""

    request_id: str
    queries: tuple[str, ...]  # usually one; a very big area is cut into several
    servers: tuple[str, ...]  # the same public servers the server download uses
    waits: tuple[float, ...]  # seconds to wait before each retry of the first server when it is busy
    busy_statuses: tuple[int, ...]
    network_type: str
    lat: float
    lon: float
    radius_m: float

    def for_browser(self) -> dict:
        """The plain data (numbers, text and lists) sent to the JavaScript."""
        return {
            "id": self.request_id,
            "format": REPLY_FORMAT,
            "queries": list(self.queries),
            "servers": [server.rstrip("/") + "/interpreter" for server in self.servers],
            "waits": list(self.waits),
            "busyStatuses": sorted(self.busy_statuses),
            "maxPayloadChars": MAX_PAYLOAD_CHARS,
        }


class NeedPhoneStreets(Exception):
    """Raised by ``get_network`` when nothing saved or ready-made covers the place and the phone should fetch it."""

    def __init__(self, request: PhoneRequest):
        super().__init__("The streets for this place have to be downloaded by the phone")
        self.request = request


def _polygons(lat: float, lon: float, radius_m: float):
    """(the area asked for, the same area 500 m wider) exactly as ``ox.graph_from_point`` makes them."""
    bbox = ox.utils_geo.bbox_from_point((lat, lon), radius_m)
    polygon = ox.utils_geo.bbox_to_poly(bbox)
    projected, crs = ox.projection.project_geometry(polygon)
    buffered, _ = ox.projection.project_geometry(projected.buffer(500), crs=crs, to_latlong=True)
    return polygon, buffered


def build_request(
    lat: float,
    lon: float,
    network_type: str,
    radius_m: float,
    servers: tuple[str, ...] = OVERPASS_SERVERS,
    waits: tuple[float, ...] = RETRY_WAITS_S,
) -> PhoneRequest:
    """The Overpass question(s) for the streets within ``radius_m`` of a point: word for word what OSMnx would send."""
    _, buffered = _polygons(lat, lon, radius_m)
    overpass_settings = _overpass._make_overpass_settings()
    way_filter = street_filter(network_type)
    queries = tuple(
        f"{overpass_settings};(way{way_filter}(poly:{coords!r});>;);out;"
        for coords in _overpass._make_overpass_polygon_coord_strs(buffered)
    )
    return PhoneRequest(
        request_id=uuid.uuid4().hex,
        queries=queries,
        servers=tuple(servers),
        waits=tuple(waits),
        busy_statuses=tuple(sorted(BUSY_STATUS_CODES)),
        network_type=network_type,
        lat=float(lat),
        lon=float(lon),
        radius_m=float(radius_m),
    )


# --- reading what the phone sent back ----------------------------------------------------------------


def _unpack(reply: dict) -> dict:
    """The phone's reply (text, maybe compressed) -> the plain dict it describes. Raises PhoneStreetsError."""
    payload = reply.get("payload")
    if not isinstance(payload, str) or not payload:
        raise PhoneStreetsError("The phone sent back an empty answer")
    if len(payload) > MAX_PAYLOAD_CHARS:
        raise PhoneStreetsError("The phone's answer was too big")
    encoding = reply.get("encoding")
    try:
        if encoding == "gzip-base64":
            raw = base64.b64decode(payload, validate=True)
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:
                text = f.read(MAX_JSON_BYTES + 1)
            if len(text) > MAX_JSON_BYTES:
                raise PhoneStreetsError("The phone's answer unpacked to something too big")
            data = json.loads(text)
        elif encoding == "json":
            data = json.loads(payload)
        else:
            raise PhoneStreetsError(f"The phone used a way of sending that we don't know ({encoding!r})")
    except (binascii.Error, OSError, EOFError, ValueError) as error:
        raise PhoneStreetsError(f"The phone's answer could not be read ({type(error).__name__})") from error
    if not isinstance(data, dict) or data.get("v") != REPLY_FORMAT:
        raise PhoneStreetsError("The phone's answer has a layout we don't know")
    return data


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _elements(data: dict) -> list[dict]:
    """Check the shape and the plausibility of the phone's lists and return them as Overpass-style elements.

    The text comes from a visitor's browser, so it is never trusted: every row is checked, positions must be real, and every street must use points that were sent.
    """
    nodes, ways = data.get("nodes"), data.get("ways")
    if not isinstance(nodes, list) or not isinstance(ways, list):
        raise PhoneStreetsError("The phone's answer is missing its streets")
    if not ways or not nodes:
        raise PhoneStreetsError("The phone didn't get any streets for that place")
    if len(nodes) > MAX_NODES:
        raise PhoneStreetsError("The phone's answer has too many points")

    known: set[int] = set()
    elements: list[dict] = []
    for row in nodes:
        if not (isinstance(row, list) and len(row) == 3 and _is_int(row[0])):
            raise PhoneStreetsError("The phone's answer has a broken point")
        node_id, y, x = row
        # Overpass returns whole streets, so a long road may reach far outside the area: only the range is checked here.
        # (Points far from the place are cut away later; if none are near it, the result is empty and is refused.)
        if not (_is_number(y) and _is_number(x)) or not (-90 <= y <= 90 and -180 <= x <= 180):
            raise PhoneStreetsError("The phone's answer has a point that is not on Earth")
        known.add(node_id)
        elements.append({"type": "node", "id": node_id, "lat": y, "lon": x})

    for row in ways:
        if not (isinstance(row, list) and len(row) == 5 and _is_int(row[0]) and isinstance(row[4], list)):
            raise PhoneStreetsError("The phone's answer has a broken street")
        way_id, highway, junction, oneway, members = row
        if not all(isinstance(tag, str) for tag in (highway, junction, oneway)) or not highway:
            raise PhoneStreetsError("The phone's answer has a street without a type")
        if len(members) < 2 or not all(_is_int(m) and m in known for m in members):
            raise PhoneStreetsError("The phone's answer has a street that uses a missing point")
        tags = {"highway": highway}
        if junction:
            tags["junction"] = junction
        if oneway:
            tags["oneway"] = oneway
        elements.append({"type": "way", "id": way_id, "nodes": members, "tags": tags})
    return elements


def build_graph(responses: Iterable[dict], network_type: str, lat: float, lon: float, radius_m: float) -> nx.MultiDiGraph:
    """The streets from Overpass-style answers, by the same steps as ``ox.graph_from_polygon`` after its download."""
    polygon, buffered = _polygons(lat, lon, radius_m)
    graph = ox.graph._create_graph(responses, network_type in ox_settings.bidirectional_network_types)
    graph = ox.truncate.truncate_graph_polygon(graph, buffered, truncate_by_edge=False)
    graph = ox.truncate.largest_component(graph, strongly=False)
    graph = ox.simplification.simplify_graph(graph)
    final = ox.truncate.truncate_graph_polygon(graph, polygon, truncate_by_edge=False)
    final = ox.truncate.largest_component(final, strongly=False)
    nx.set_node_attributes(final, values=ox.stats.count_streets_per_node(graph, nodes=final.nodes), name="street_count")
    return final


def graph_from_reply(reply: dict, lat: float, lon: float, network_type: str, radius_m: float) -> nx.MultiDiGraph:
    """Turn the phone's reply into a street network. Raises PhoneStreetsError if the phone failed or sent rubbish.

    A reply is a dict: ``{"ok": False, "error": "..."}`` when the phone gave up, or
    ``{"ok": True, "encoding": "gzip-base64" | "json", "payload": "..."}``.
    """
    if not isinstance(reply, dict):
        raise PhoneStreetsError("The phone sent no answer")
    if not reply.get("ok"):
        raise PhoneStreetsError(str(reply.get("error") or "The phone could not download the streets")[:300])
    elements = _elements(_unpack(reply))
    try:
        graph = build_graph([{"elements": elements}], network_type, lat, lon, radius_m)
    except PhoneStreetsError:
        raise
    except Exception as error:  # OSMnx says "no data" or "empty graph" in several ways
        raise PhoneStreetsError(f"The streets from the phone could not be used ({type(error).__name__})") from error
    if graph.number_of_edges() == 0:
        raise PhoneStreetsError("The phone didn't get any streets for that place")
    return graph


def encode_reply(
    elements_by_query: Iterable[Iterable[dict]], *, compress: bool = True
) -> dict:
    """Make a reply the way the JavaScript does (for tests and for trying things out without a phone).

    ``elements_by_query``: for each query, the "elements" list of Overpass's JSON answer.
    """
    nodes: dict[int, list] = {}
    ways: dict[int, list] = {}
    for elements in elements_by_query:
        for element in elements:
            if element["type"] == "node":
                nodes[element["id"]] = [element["id"], element["lat"], element["lon"]]
            elif element["type"] == "way":
                tags = element.get("tags", {})
                ways[element["id"]] = [
                    element["id"], tags.get("highway", ""), tags.get("junction", ""), tags.get("oneway", ""), element["nodes"]
                ]
    text = json.dumps({"v": REPLY_FORMAT, "nodes": list(nodes.values()), "ways": list(ways.values())}, separators=(",", ":"))
    if compress:
        return {"ok": True, "encoding": "gzip-base64", "payload": base64.b64encode(gzip.compress(text.encode())).decode()}
    return {"ok": True, "encoding": "json", "payload": text}

