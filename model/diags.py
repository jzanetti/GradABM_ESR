from logging import getLogger
from os import listdir, makedirs
from os.path import exists, join
from pickle import dump as pickle_dump
from pickle import load as pickle_load

from geopandas import GeoDataFrame, GeoSeries
from matplotlib.pyplot import (
    close,
    fill_between,
    legend,
    plot,
    savefig,
    scatter,
    subplots,
    tight_layout,
    title,
    xlabel,
    xlim,
    ylabel,
    ylim,
    yscale,
)
from numpy import array, count_nonzero, max, min, random
from pandas import Series, read_csv
from PIL import Image
from pyproj import CRS, Transformer
from shapely.geometry import Point
from shapely.wkt import loads
from torch import load as torch_load
from torch import save as torch_save

from model import STAGE_INDEX

logger = getLogger()


def save_outputs(param_model, workdir):
    if not exists(workdir):
        makedirs(workdir)

    torch_save(param_model["param_model"], join(workdir, "param_model.model"))
    pickle_dump(param_model["params"], open(join(workdir, "params.p"), "wb"))
    pickle_dump(param_model["output_info"], open(join(workdir, "output_info.p"), "wb"))
    # pickle_dump(param_model["agents"], open(join(workdir, "output_agents.p"), "wb"))


def load_outputs(param_path: str, output_info_path: str, param_model_path: str):
    param = pickle_load(open(param_path, "rb"))
    output_info = pickle_load(open(output_info_path, "rb"))
    param_model = torch_load(param_model_path)
    # output = pickle_load(open("output.p", "rb"))
    return {"param": param, "output_info": output_info, "param_model": param_model}


def obtain_sa2_info(sa2, target_areas, target_attrs):
    def _random_point_within_polygon(polygon):
        min_x, min_y, max_x, max_y = polygon.bounds
        while True:
            point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if polygon.contains(point):
                return point.y, point.x

    outputs = []
    for sa2_code in target_areas:
        try:
            row = sa2[sa2["SA22018_V1_00"] == sa2_code].iloc[0]  # Get the corresponding row from y
        except IndexError:
            continue
        polygon = loads(row["WKT"])  # Parse the WKT to a polygon
        lat, lon = _random_point_within_polygon(polygon)
        outputs.append((lat, lon))

    return outputs


def plot_diags(
    workdir: str,
    outputs,
    epoch_loss_lists,
    vis_cfg,
    apply_log_for_loss: bool = False,
    start_timestep: int or None = None,
    end_timestep: int or None = None,
):
    if not exists(workdir):
        makedirs(workdir)

    for i, output in enumerate(outputs):
        my_pred = output["pred"].tolist()

        # ----------------------------
        # Plot agents
        # ----------------------------
        susceptible_counts = count_nonzero(
            output["all_records"] == STAGE_INDEX["susceptible"], axis=1
        )
        exposed_counts = count_nonzero(output["all_records"] == STAGE_INDEX["exposed"], axis=1)
        infected_counts = count_nonzero(output["all_records"] == STAGE_INDEX["infected"], axis=1)
        recovered_or_death_counts = count_nonzero(
            output["all_records"] == STAGE_INDEX["recovered_or_death"], axis=1
        )

        if i == 0:
            plot(susceptible_counts, label="Susceptible", color="c")
            plot(exposed_counts, label="Exposed", color="g")
            plot(infected_counts, label="Infected", color="r")
            plot(recovered_or_death_counts, label="Recovery + Death", color="b")
            plot(my_pred, label="Death", color="m")
        else:
            plot(susceptible_counts, color="c")
            plot(exposed_counts, color="g")
            plot(infected_counts, color="r")
            plot(recovered_or_death_counts, color="b")
            plot(my_pred, color="m")

    xlabel(vis_cfg["temporal_res"])
    ylabel("Number of agents")
    title("Agent symptom")
    legend()
    tight_layout()
    savefig(join(workdir, "Agents.png"), bbox_inches="tight")
    close()

    # ----------------------------
    # Plot agents map
    # ----------------------------
    if vis_cfg["agents_map"]["enable"]:
        tmp_dir = join(workdir, "tmp")
        if not exists(tmp_dir):
            makedirs(tmp_dir)
        sa2 = read_csv(vis_cfg["agents_map"]["sa2_path"])
        sa2 = sa2.loc[sa2["LAND_AREA_SQ_KM"] > 0]
        gdf = GeoDataFrame(sa2, geometry=GeoSeries.from_wkt(sa2["WKT"]))
        gdf = gdf.cx[
            vis_cfg["agents_map"]["domain"]["min_lon"] : vis_cfg["agents_map"]["domain"][
                "max_lon"
            ],
            vis_cfg["agents_map"]["domain"]["min_lat"] : vis_cfg["agents_map"]["domain"][
                "max_lat"
            ],
        ]
        proc_latlons = []
        proc_attrs = []
        for t in range(vis_cfg["agents_map"]["start_t"], len(output["all_target_indices"])):
            proc_indices = output["all_target_indices"][t]
            if len(proc_indices) == 0:
                continue

            # num_to_select = int(len(proc_indices) * random.uniform(0.1, 0.3))
            # from random import sample as random_sample

            # Randomly select items
            # proc_indices = random_sample(list(proc_indices), num_to_select)

            proc_latlon = obtain_sa2_info(
                sa2, array(output["agents_area"])[proc_indices], output["agents_ethnicity"]
            )
            proc_attr = array(output["agents_ethnicity"])[proc_indices]

            scatter_colors = {0: "b", 1: "g", 2: "r", 3: "c", 4: "m"}
            colors = [scatter_colors[val] for val in proc_attr]

            proc_latlons.extend(proc_latlon)
            proc_attrs.extend(colors)

            lats = [lat for lat, _ in proc_latlons]
            lons = [lon for _, lon in proc_latlons]
            gdf.plot(figsize=(10, 10))

            scatter(lons, lats, color=proc_attrs, marker="x")
            tight_layout()
            title(f"{t}, Total agents: {len(proc_latlons)}")
            savefig(join(tmp_dir, f"agents_{t}.png"), bbox_inches="tight")
            close()

        png_files = [f for f in listdir(tmp_dir) if f.endswith(".png")]
        png_files = sorted(png_files, key=lambda item: int(item.split("_")[1].split(".")[0]))
        images = []

        # Open each PNG image and append to the list
        for png_file in png_files:
            image_path = join(tmp_dir, png_file)
            img = Image.open(image_path)
            images.append(img)

        if len(images) > 0:
            images[0].save(
                join(workdir, "agents.gif"),
                format="GIF",
                save_all=True,
                append_images=images[1:],
                duration=1000,  # Duration between frames in milliseconds
                loop=0,  # 0 means infinite loop
            )
        else:
            logger.info("Not able to get agents on the map to create gif")

    # ----------------------------
    # Plot losses
    # ----------------------------
    for epoch_loss_list in epoch_loss_lists:
        plot(epoch_loss_list, "k")

    if apply_log_for_loss:
        yscale("log")
    xlabel("Epoch")
    ylabel("Loss")
    title("Loss")
    tight_layout()
    savefig(join(workdir, "loss.png"), bbox_inches="tight")
    close()

    # ----------------------------
    # Plot Prediction/Truth
    # ----------------------------
    all_preds = []
    _, ax = subplots()
    for i, output in enumerate(outputs):
        my_pred = output["pred"].tolist()
        if i == 0:
            my_targ = output["y"].tolist()
            plot(my_targ, color="k", linewidth=2.0, label="Observed cases")

        if len(outputs) == 1:
            plot(my_pred, linewidth=1.0, linestyle="--")
        else:
            all_preds.append(my_pred)

    if len(all_preds) > 0:
        all_preds = array(all_preds)
        y_min = min(all_preds, 0)
        y_max = max(all_preds, 0)
        x = range(len(y_min))
        ax.fill_between(x, y_min, y_max, alpha=0.5)

    if start_timestep is not None and end_timestep is not None:
        # Set the x-axis tick positions and labels
        xtick_positions = list(range(start_timestep, end_timestep))
        tick_labels = [str(tick) for tick in xtick_positions]
        ax.set_xticks(x[::3])
        ax.set_xticklabels(tick_labels[::3])

    legend()
    title(
        f"JUNE-NZ validation {vis_cfg['name']} \n Simulation: {int(sum(my_pred))} vs Observed: {int(sum(my_targ))}"
    )
    xlabel(f"{vis_cfg['temporal_res']}")
    ylabel("Number of cases")
    tight_layout()
    savefig(join(workdir, "prediction_vs_truth.png"), bbox_inches="tight")
    close()
