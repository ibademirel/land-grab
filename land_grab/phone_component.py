"""The Streamlit side of the phone download: mounts the small JavaScript helper (``phone_streets.mjs``).

Kept out of ``app.py`` because a Streamlit component must be registered once, not on every rerun of the page,
and a module is only run once.
"""

from pathlib import Path

import streamlit as st

from land_grab.browser_streets import PhoneRequest

COMPONENT_KEY = "phone_streets"
_JS = (Path(__file__).with_name("phone_streets.mjs")).read_text(encoding="utf-8")
_component = st.components.v2.component("land_grab_phone_streets", js=_JS)


def ask_phone(request: PhoneRequest) -> dict | None:
    """Show the helper (it downloads on the phone and writes its progress) and return the phone's reply when it has come.

    The reply is ``{"id", "ok", "encoding", "payload"}`` or ``{"id", "ok": False, "error"}``, or None while the
    phone is still working. It may be an old one: check that its ``id`` is the request's.
    """
    result = _component(key=COMPONENT_KEY, data=request.for_browser(), on_reply_change=lambda: None)
    reply = result.reply
    return reply if isinstance(reply, dict) else None
