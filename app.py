"""Land Grab Route Planner: a friendly web page for planning a hexagon-claiming loop.

    .venv/bin/python -m streamlit run app.py

This file is only the screen. The work (planner-only routing, the cache of
finished routes, the plain-English errors) is in ``land_grab/web.py``.
"""

import streamlit as st

from land_grab.web import ACTIVITIES, MAX_KM, STAGE_BUSY, details_text, friendly_error, get_route

ACTIVITY_LABELS = {"run": "🏃 Run", "walk": "🚶 Walk", "ride": "🚴 Ride"}
MAP_HEIGHT_PX = 560  # the CSS below makes it taller on big screens and keeps it on-screen on phones

CSS = """
<style>
/* Hide Streamlit's own menu, toolbar and footer. */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stMainMenu"],
[data-testid="stDeployButton"], [data-testid="stAppDeployButton"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; visibility: hidden !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; }

/* A comfortable page: not too wide for the inputs, nearly full width for the map. */
.block-container { max-width: 1200px; padding: 2rem 1rem 3rem; }
.lg-title { margin: 0 0 .25rem; font-size: 2.4rem; line-height: 1.15; font-weight: 700; color: #1f4e5f; }
.lg-intro { margin: 0 0 1.25rem; font-size: 1.15rem; color: #4a5f66; }

/* The input block. */
[data-testid="stForm"] { max-width: 640px; border-radius: 1rem; padding: 1.25rem; }
[data-testid="stForm"] input { font-size: 1.15rem; min-height: 3rem; }
[data-testid="stForm"] label p { font-size: 1.05rem; font-weight: 600; }

/* Run / Walk / Ride as three large buttons. */
[data-testid="stButtonGroup"] { width: 100%; }
[data-testid="stButtonGroup"] button { flex: 1 1 0; min-height: 3.5rem; font-size: 1.15rem; justify-content: center; }
[data-testid="stFormSubmitButton"] button { width: 100%; min-height: 3.5rem; font-size: 1.2rem; font-weight: 600; }

/* The two number cards. */
.st-key-cards [data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #cfe0e0; border-radius: 1rem; padding: 1rem 1.25rem; text-align: center;
}
.st-key-cards [data-testid="stMetricLabel"] { justify-content: center; font-size: 1.05rem; color: #4a5f66; }
.st-key-cards [data-testid="stMetricValue"] { justify-content: center; font-size: 2rem; line-height: 1.15; color: #1f4e5f; }
.st-key-cards [data-testid="stMetricValue"] > div { white-space: normal; overflow: visible; text-overflow: clip; text-align: center; }

/* The map: as wide as the page, and tall enough to use. */
.st-key-map iframe { width: 100% !important; height: 70vh !important; min-height: 380px; border-radius: 1rem; border: 1px solid #cfe0e0; }
.st-key-download button { width: 100%; min-height: 3.25rem; font-size: 1.1rem; }
[data-testid="stExpander"] summary { font-size: 1.05rem; min-height: 3rem; }

/* Phones: one column, big tap targets, nothing cut off. */
@media (max-width: 640px) {
    .block-container { padding: 1rem .6rem 2rem; }
    .lg-title { font-size: 1.8rem; }
    .lg-intro { font-size: 1.05rem; }
    [data-testid="stForm"] { max-width: 100%; padding: .9rem; }
    /* The two cards sit one above the other, each as wide as the screen, so "17 hexagons = 17 points" never gets cut. */
    .st-key-cards [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: .5rem; }
    .st-key-cards [data-testid="stColumn"] { min-width: 100% !important; flex: 1 1 100% !important; width: 100% !important; }
    .st-key-cards [data-testid="stMetric"] { padding: .6rem .5rem; }
    .st-key-cards [data-testid="stMetricLabel"] { font-size: .9rem; }
    .st-key-cards [data-testid="stMetricValue"] { font-size: 1.6rem; }
    .st-key-map iframe { height: 60vh !important; }
}
</style>
"""


def show_error(error: Exception, postcode: str) -> None:
    st.error(friendly_error(error, postcode))
    with st.expander("Details", expanded=False):  # the technical message, for the curious
        st.code(f"{type(error).__name__}: {error}")


def plan(postcode: str, km: float, activity: str, claimed: str = "") -> None:
    """Make (or fetch) the route, showing each step, and keep the result for the next rerun."""
    st.session_state.pop("result", None)
    with st.status("Getting started", expanded=True) as status:

        def stage(name: str) -> None:
            status.update(label=name)
            st.write(f"⏳ {name}" if name == STAGE_BUSY else f"✔ {name}")  # a retry is not a finished step

        try:
            result, from_cache = get_route(postcode, km, activity, claimed, on_stage=stage)
        except Exception as error:  # every failure becomes a friendly message, never a traceback
            status.update(label="That didn't work", state="error", expanded=False)
            st.session_state["error"] = (error, postcode)
            return
        status.update(
            label="Here it is, straight from your earlier search" if from_cache else "Your route is ready",
            state="complete",
            expanded=False,
        )
    st.session_state.pop("error", None)
    st.session_state["result"] = (result, from_cache)


st.set_page_config(page_title="Land Grab Route Planner", page_icon="🗺️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<h1 class="lg-title">Land Grab Route Planner</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="lg-intro">Type a UK postcode and how far you want to go, and we\'ll plan a loop '
    "that claims as many hexagons as it can.</p>",
    unsafe_allow_html=True,
)

with st.form("route_form", border=True):
    postcode = st.text_input("Postcode", value="SW1A 1AA", placeholder="for example SW1A 1AA")
    km = st.number_input(
        "How far? (km)",
        min_value=1.0,
        max_value=max(MAX_KM.values()),
        value=5.0,
        step=1.0,
        help="Longest here: " + ", ".join(f"{limit:g} km {name}" for name, limit in MAX_KM.items()) + ".",
    )
    activity = st.segmented_control(
        "Activity",
        list(ACTIVITIES),
        default="run",
        required=True,
        format_func=ACTIVITY_LABELS.get,
        width="stretch",
    )
    claimed = st.text_area(
        "Cells already claimed (yours or rivals'), optional",
        value="",
        placeholder="H3 cell IDs separated by commas, for example 88194ad147fffff, 88194ad141fffff",
        help="Paste the cell IDs from the challenge map, whether you or a rival own them. "
        "They show in purple, are not counted as new, and the route avoids them where it can.",
        height=100,
    )
    submitted = st.form_submit_button("Plan my route", type="primary")

if submitted:
    plan(postcode, float(km), activity or "run", claimed)

if "error" in st.session_state and "result" not in st.session_state:
    show_error(*st.session_state["error"])

if "result" in st.session_state:
    result, from_cache = st.session_state["result"]

    with st.container(key="cards"):
        hexagons_col, distance_col = st.columns(2)
        hexagons_col.metric("Hexagons", f"{result.hexagons} hexagons = {result.hexagons} points")
        distance_col.metric("Distance", f"{result.distance_km:.1f} km")

    for problem in result.loop.problems:
        st.warning(str(problem))

    with st.container(key="map"):
        st.iframe(result.map_html, height=MAP_HEIGHT_PX)  # our own map page, so embedding the HTML is safe

    with st.container(key="download"):
        st.download_button(
            "⬇️ Download GPX for your watch or phone",
            data=result.gpx,
            file_name="land_grab_route.gpx",
            mime="application/gpx+xml",
            type="primary",
        )

    with st.expander("Details", expanded=False):
        st.code(details_text(result, from_cache))
