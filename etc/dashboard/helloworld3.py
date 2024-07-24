import folium
import pandas as pd
from folium.plugins import TimestampedGeoJson

# Create a DataFrame
df = pd.DataFrame(
    {
        "time": pd.date_range(start="2024-01-01", periods=4).astype(str),
        "lat": [-33.45, 37.75, 28.61, 39.91],
        "lon": [-70.67, -122.45, 77.23, 116.39],
    }
)

# Convert DataFrame to GeoJSON
data = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": {
                "times": [time],
                "style": {"color": "red"},
                "icon": "circle",
                "iconstyle": {
                    "fillColor": "red",
                    "fillOpacity": 0.6,
                    "stroke": "false",
                    "radius": 5,
                },
            },
        }
        for time, lat, lon in zip(df["time"], df["lat"], df["lon"])
    ],
}

# Create a map
m = folium.Map([0, 0], zoom_start=2)

# Add TimestampedGeoJson to the map
TimestampedGeoJson(
    data,
    period="P1D",
    add_last_point=True,
).add_to(m)

# Display the map
m.save("test.html")
