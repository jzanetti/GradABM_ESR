from pickle import load as pickle_load

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output

df = pickle_load(open("etc/dashboard/testdata/extracted_data.pickle", "rb"))[
    "sampled_data"
]
app = dash.Dash(__name__)

fig = px.scatter_mapbox(
    df,
    lat="latitude",
    lon="longitude",
    color="type",
    animation_frame="hour",
    color_continuous_scale=px.colors.cyclical.IceFire,
    size_max=15,
    zoom=10,
)

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

app.layout = html.Div(
    [
        dcc.Graph(id="map", figure=fig),
        html.Label("Time Slider"),
        dcc.Slider(
            id="time-slider",
            min=df["t"].min(),
            max=df["t"].max(),
            value=df["t"].min(),
            marks={str(t): str(t) for t in df["t"].unique()},
            step=None,
        ),
    ]
)


@app.callback(Output("map", "figure"), [Input("time-slider", "value")])
def update_figure(selected_t):
    filtered_df = df[df.t == selected_t]
    fig = px.scatter_mapbox(
        filtered_df,
        lat="lat",
        lon="lon",
        color="type",
        color_continuous_scale=px.colors.cyclical.IceFire,
        size_max=15,
        zoom=10,
    )
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
