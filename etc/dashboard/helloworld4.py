import dash
import dash_html_components as html
import dash_leaflet as dl

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        dl.Map(
            center=[56.05, 10.25],
            zoom=10,
            children=[
                dl.TileLayer(),
                dl.Marker(position=[56.05, 10.25], children=[dl.Tooltip("Aarhus")]),
            ],
            style={
                "width": "100%",
                "height": "50vh",
                "margin": "auto",
                "display": "block",
            },
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
