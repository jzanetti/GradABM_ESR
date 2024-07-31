# read output:
from pickle import load as pickle_load

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import imageio
import matplotlib.pyplot as plt
from pandas import merge, read_parquet

from process.train_wrapper import train
from process.utils.utils import setup_logging

workdir = "/tmp/gradabm_esr/test"

logger = setup_logging(workdir)

# ---------------------------
# Train the model
# ---------------------------
train(workdir, use_test_data=True)

# ---------------------------
# Create gif of the trained model
# ---------------------------
output_path = f"{workdir}/output_info.p"
data = pickle_load(open(output_path, "rb"))
all_records = data["all_records"]
pred_indices = data["pred_indices"]

location = read_parquet(
    "/DSC/digital_twin/abm/PHA_report_202405/syspop/NZ/2023/median/syspop_location.parquet"
)
location = location[location["type"] == "household"]
location["area"] = location["name"].apply(lambda x: x.split("_")[0])

# Group Y by 'area' and randomly select one row from each group
location_grouped = location.groupby("area").apply(
    lambda x: x.sample(1, random_state=999)
)
location_grouped = location_grouped[["area", "latitude", "longitude"]]
location_grouped["area"] = location_grouped["area"].astype("int")
all_records["area"] = all_records["area"].astype("int")

# Reset the index of Y_grouped
location_grouped.reset_index(drop=True, inplace=True)
all_records = merge(all_records, location_grouped, on="area", how="left")

status_dict = {
    # 2: {"status": "exposed", "color": "yellow", "marker": "x", "size": 15},
    4: {"status": "infected", "color": "red", "marker": "x", "size": 15},
    # 8: {"status": "recovered_or_death", "color": "green", "marker": "o", "size": 10},
}


min_longitude = 174.59  # all_records["longitude"].min() - 0.05
max_longitude = 174.9  # all_records["longitude"].max() + 0.05

min_latitude = -37.01  # all_records["latitude"].min() - 0.05
max_latitude = -36.73  # all_records["latitude"].max() + 0.05
stamen_terrain = cimgt.OSM()

color_map = {
    "European": "r",
    "Maori": "b",
    "Pacific": "c",
    "Asian": "m",
    "MELAA": "k",
    # Add more ethnicities and colors as needed
}

ethnicity_map = {0: "European", 1: "Maori", 2: "Pacific", 3: "Asian", 4: "MELAA"}
start_i = 3
filenames = []
for t in range(start_i, 26):
    print(t)
    proc_indices = pred_indices[t]
    proc_indices_accum = [
        item for sublist in pred_indices[start_i : t + 1] for item in sublist
    ]
    proc_data = all_records[[t, "latitude", "longitude", "ethnicity"]]
    proc_data["ethnicity"] = proc_data["ethnicity"].replace(ethnicity_map)
    # Create a new figure
    fig = plt.figure(figsize=(10, 5))

    # Create a map projection
    ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)

    # Add coastlines
    # ax.coastlines()
    ax.set_extent([min_longitude, max_longitude, min_latitude, max_latitude])
    ax.add_image(stamen_terrain, 10)
    # Plot dots for each status
    df_filtered = proc_data.loc[proc_indices]
    df_filtered_accum = proc_data.loc[proc_indices_accum]

    ax.scatter(
        df_filtered_accum["longitude"],
        df_filtered_accum["latitude"],
        color="grey",
        marker="o",
        s=30,
        alpha=0.3,
        transform=ccrs.PlateCarree(),
    )

    for ethnicity in df_filtered["ethnicity"].unique():

        df_ethnicity2 = df_filtered[df_filtered["ethnicity"] == ethnicity]
        ax.scatter(
            df_ethnicity2["longitude"],
            df_ethnicity2["latitude"],
            color=color_map[ethnicity],
            marker="o",
            s=60,
            alpha=0.7,
            label=ethnicity,
            transform=ccrs.PlateCarree(),
        )
    # Add a legend
    ax.legend()

    ax.set_title(f"Time Step {t}")

    # Show the plot
    filename = f"frames/{t}.png"
    filenames.append(filename)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()

# Create a list to store the images
import imageio

images = []

# Read each file and append it to the images list
for filename in filenames:
    images.append(imageio.imread(filename))

# Save the images as a GIF with a duration of 1 second between each frame
imageio.mimsave("frames/output.gif", images, duration=1000)
