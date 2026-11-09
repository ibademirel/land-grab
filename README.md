# land_grab

1. land_grab turns a GPS track (GPX) into a claimed "territory" on a map.
2. It reads GPX routes with gpxpy and projects them to metric coordinates with pyproj.
3. It uses shapely to build polygons from closed loops and measure the area captured.
4. It downloads street networks from OpenStreetMap with osmnx and networkx to snap and analyse routes.
5. Results are rendered as interactive maps with folium; tests run with pytest.

## Run the web app

```
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

To put it online for friends, see `DEPLOY.md`. The tests need `requirements-dev.txt` (it adds pytest).
