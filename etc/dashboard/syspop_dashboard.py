from os.path import exists
from pickle import dump as pickle_dump
from pickle import load as pickle_load

import dash
import dash_core_components as dcc
import dash_html_components as html
import folium
import pandas as pd
from dash.dependencies import Input, Output
from folium.plugins import HeatMap, TimestampedGeoJson
from funcs.func_diary import get_diary_data
from funcs.func_io import read_diaries, read_syspop
from funcs.func_util import perturbate_latlon

app = dash.Dash(__name__)

# ------------------------------
# Read the data
# ------------------------------
preprocessed_data_path = "etc/dashboard/testdata/preproc_data.pickle"
force_preproc_data = False


if force_preproc_data or (not exists(preprocessed_data_path)):
    syspop_data = read_syspop(
        syspop_path="etc/dashboard/testdata/syspop_base.csv",
        syspop_address_path="etc/dashboard/testdata/syspop_location.csv",
        sample_size=None,
    )
    diary_data = read_diaries(diary_path="etc/dashboard/testdata/diaries.parquet")
    diary_location = get_diary_data(
        syspop_data["base"], syspop_data["address"], diary_data
    )

    diary_location = perturbate_latlon(diary_location)

    syspop_address_data = syspop_data["address"]

    pickle_dump(
        {"diary_location": diary_location, "syspop_address_data": syspop_address_data},
        open(preprocessed_data_path, "wb"),
    )

else:
    preproc_data = pickle_load(open(preprocessed_data_path, "rb"))
    diary_location = preproc_data["diary_location"]
    syspop_address_data = preproc_data["syspop_address_data"]


# Define the layout
app.layout = html.Div(
    [
        html.Iframe(id="map", width="100%", height="600"),
        html.Div(
            [
                html.Label("Data type"),
                dcc.Dropdown(
                    id="type-dropdown",
                    options=[
                        {"label": "Synthetic population (density)", "value": "syspop"},
                        {"label": "Synthetic population (plan)", "value": "diary"},
                    ],
                    value="diary",
                    style={"width": "50%"},
                ),
                html.Label("Additional Options"),
                dcc.Dropdown(
                    id="place-dropdown",
                    value="company",
                    style={"width": "50%"},
                ),
                html.Label("Data fraction"),
                dcc.Dropdown(
                    id="frac-dropdown",
                    value=1000,
                    style={"width": "50%"},
                ),
                html.Label("Color indicator"),
                dcc.Dropdown(
                    id="color-indicator-dropdown",
                    value="ethnicity",
                    style={"width": "50%"},
                ),
            ],
            style={"display": "inline-block", "width": "50%"},
        ),
        html.Div(
            [
                html.Button("Generate Plot", id="generate-plot", n_clicks=0),
                dcc.Graph(id="dynamic-graph"),
            ]
        ),
    ]
)


import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


@app.callback(Output("dynamic-graph", "figure"), [Input("generate-plot", "n_clicks")])
def update_graph(n_clicks):
    if n_clicks > 0:
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        fig = go.Figure(data=go.Scatter(x=x, y=y))
        return fig
    else:
        # Return an empty figure
        return go.Figure()


# Define the callback
@app.callback(
    Output("color-indicator-dropdown", "options"), Input("type-dropdown", "value")
)
def update_second_dropdown(value):
    if value == "syspop":
        return []
    elif value == "diary":
        return [
            {"label": "Ethnicity", "value": "ethnicity"},
            {"label": "Age", "value": "age"},
        ]


# Define the callback
@app.callback(Output("place-dropdown", "options"), Input("type-dropdown", "value"))
def update_second_dropdown(value):
    if value == "syspop":
        return [
            {"label": "Household", "value": "household"},
            {"label": "Supermarket", "value": "supermarket"},
            {"label": "company", "value": "company"},
            {"label": "school", "value": "school"},
            {"label": "hospital", "value": "hospital"},
            {"label": "restaurant", "value": "restaurant"},
        ]
    elif value == "diary":
        return [
            {"label": "Household", "value": "household"},
            {"label": "Supermarket", "value": "supermarket"},
            {"label": "company", "value": "company"},
            {"label": "school", "value": "school"},
            {"label": "hospital", "value": "hospital"},
            {"label": "restaurant", "value": "restaurant"},
            {"label": "all", "value": "all"},
        ]


# Define the callback
@app.callback(Output("frac-dropdown", "options"), Input("type-dropdown", "value"))
def update_second_dropdown(value):
    if value == "syspop":
        return [
            {"label": "10%", "value": 0.1},
            {"label": "30%", "value": 0.3},
            {"label": "50%", "value": 0.5},
            {"label": "100%", "value": 1.0},
        ]
    elif value == "diary":
        return [
            {"label": 1000, "value": 1000},
            {"label": 5000, "value": 5000},
            {"label": 10000, "value": 10000},
            {"label": 30000, "value": 30000},
            {"label": 100000, "value": 100000},
        ]


# Define the callback
@app.callback(
    Output("map", "srcDoc"),
    [
        Input("type-dropdown", "value"),
        Input("place-dropdown", "value"),
        Input("frac-dropdown", "value"),
        Input("color-indicator-dropdown", "value"),
    ],
)
def update_map(selected_type, selected_place, selected_frac, selected_color_indicator):
    if selected_type == "syspop":
        df = syspop_address_data[syspop_address_data["type"] == selected_place].sample(
            frac=selected_frac
        )
        m = folium.Map(
            location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=13
        )
        HeatMap(data=df[["latitude", "longitude"]], radius=8, max_zoom=13).add_to(m)
    elif selected_type == "diary":
        if selected_place != "all":
            df = diary_location[diary_location["type"] == selected_place]
        df = df.sample(selected_frac)
        df = perturbate_latlon(df, perturbation_range=0.0001)

        m = folium.Map(
            location=[
                df["latitude"].mean(),
                df["longitude"].mean(),
            ],
            zoom_start=13,
        )

        df["timestamp"] = pd.to_datetime(df["hour"], format="%H").apply(
            lambda dt: dt.replace(year=2022, month=1, day=1)
        )

        if selected_color_indicator == "ethnicity":
            legend_colors = {
                "European": "#FF0000",
                "Asian": "#37ff00",
                "Maori": "#00fff7",
                "Pacific": "#002aff",
                "MELAA": "#ff00dd",
            }

        df["timestamp"] = df["timestamp"].astype(str)
        features = []
        diary_map = {}
        for (
            lat,
            lon,
            timestamp,
            data_type,
            data_age,
            data_gender,
            data_ethnicity,
            data_id,
        ) in zip(
            df["latitude"],
            df["longitude"],
            df["timestamp"],
            df["type"],
            df["age"],
            df["gender"],
            df["ethnicity"],
            df["id"],
        ):
            if selected_color_indicator == "ethnicity":
                color_indicator = data_ethnicity

            features.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat],
                    },
                    "properties": {
                        "times": [timestamp],
                        "icon": "circle",
                        "iconstyle": {
                            "color": None,
                            "fillColor": legend_colors[color_indicator],
                            "fillOpacity": 0.5,
                            "stroke": "false",
                            "radius": 5,
                        },
                        "popup": f"- Name: {data_id} <br> - Age {data_age} <br> - Gender: {data_gender} <br> - Ethnicity: {data_ethnicity}",
                        "id": "f{data_id}",
                    },
                }
            )

            # if data_id not in diary_map:
            #    diary_map[data_id] =

        # Add the TimestampedGeoJson to the map
        TimestampedGeoJson(
            {"type": "FeatureCollection", "features": features},
            period="PT1H",
            duration="PT3H",
            add_last_point=False,
        ).add_to(m)

    return m._repr_html_()


if __name__ == "__main__":
    app.run_server(debug=False)
