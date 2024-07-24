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
prediction_path = "/tmp/gradabm_esr/policy_paper/predict/base_exp/output/pred_0_1.p"
prediction_data = pickle_load(open(prediction_path, "rb"))

agents_status = prediction_data["output"]["all_records"]

timesteps, ids = agents_status.shape

# Create a DataFrame from the array
df = pd.DataFrame(agents_status)

# Convert the DataFrame to a long format
df = df.unstack().reset_index()

# Rename the columns
df.columns = ["id", "timestep", "status"]

# Adjust the 'id' and 'timestep' to be 0-indexed, if necessary
df["id"] += 1
df["timestep"] += 1

# Define the layout
app.layout = html.Div(
    [
        html.Iframe(id="map", width="100%", height="600"),
    ]
)


# Define the callback
@app.callback(
    Output("map", "srcDoc"),
)
def update_map():
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
                    # "popup": create_popup(data_id, 15, 15),
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
