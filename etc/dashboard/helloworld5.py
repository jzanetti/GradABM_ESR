import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output, State

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


df = pd.read_csv(
    "https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv"
)
years = list(set(df["year"]))
years.sort()


def make_fig(year):
    return px.scatter(
        df[df.year == year],
        x="gdpPercap",
        y="lifeExp",
        size="pop",
        color="continent",
        hover_name="country",
        log_x=True,
        size_max=55,
    )


app.layout = html.Div(
    [
        dcc.Interval(id="animate", disabled=True),
        dcc.Graph(id="graph-with-slider", figure=make_fig(1952)),
        dcc.Slider(
            id="year-slider",
            min=df["year"].min(),
            max=df["year"].max(),
            value=df["year"].min(),
            marks={str(year): str(year) for year in df["year"].unique()},
            step=None,
        ),
        html.Button("Play", id="play"),
    ]
)


@app.callback(
    Output("graph-with-slider", "figure"),
    Output("year-slider", "value"),
    Input("animate", "n_intervals"),
    Input("year-slider", "value"),
    prevent_initial_call=True,
)
def update_figure(n, selected_year):
    index = years.index(selected_year)
    index = (index + 1) % len(years)
    year = years[index]
    return make_fig(year), year


@app.callback(
    Output("animate", "disabled"),
    Input("play", "n_clicks"),
    State("animate", "disabled"),
)
def toggle(n, playing):
    if n:
        return not playing
    return playing


if __name__ == "__main__":
    app.run_server(debug=True)
