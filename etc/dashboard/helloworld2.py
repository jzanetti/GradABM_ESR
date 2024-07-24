import dash
import dash_core_components as dcc
import dash_html_components as html
import folium
import pandas as pd
from dash.dependencies import Input, Output

# Assuming your DataFrame is named df
df = pd.DataFrame(
    {
        "type": ["A", "B", "A", "B"],
        "lat": [-37.81, -37.82, -37.83, -37.84],
        "lon": [144.96, 144.97, 144.98, 144.99],
        "time": pd.date_range(start="1/1/2022", periods=4),
    }
)

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.Button("Plot Points", id="plot-button"),
        dcc.RangeSlider(
            id="time-slider",
            min=df["time"].min().value // 10**9,
            max=df["time"].max().value // 10**9,
            value=[
                df["time"].min().value // 10**9,
                df["time"].max().value // 10**9,
            ],
            marks={
                i: str(pd.to_datetime(i, unit="s").date())
                for i in range(
                    df["time"].min().value // 10**9,
                    df["time"].max().value // 10**9 + 1,
                    86400,
                )
            },
        ),
        html.Iframe(id="map", srcDoc=None, width="100%", height="600"),
    ]
)


@app.callback(
    Output("map", "srcDoc"),
    [Input("plot-button", "n_clicks"), Input("time-slider", "value")],
)
def update_map(n, time_range):
    if n is None:
        return None

    # Filter the DataFrame based on the selected time range
    df_filtered = df[
        (df["time"] >= pd.to_datetime(time_range[0], unit="s"))
        & (df["time"] <= pd.to_datetime(time_range[1], unit="s"))
    ]

    # Create a map centered around the average latitude and longitude values
    m = folium.Map(
        location=[df_filtered["lat"].mean(), df_filtered["lon"].mean()], zoom_start=13
    )

    # Define a color dictionary for different types
    colors = {"A": "red", "B": "blue"}

    # Add points to the map
    for _, row in df_filtered.iterrows():
        folium.Marker(
            [row["lat"], row["lon"]], icon=folium.Icon(color=colors[row["type"]])
        ).add_to(m)

    # Return the map as an HTML string
    return m._repr_html_()


if __name__ == "__main__":
    app.run_server(debug=True)
