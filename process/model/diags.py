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
from numpy import NaN, array, bincount, count_nonzero, isnan, max, min, random, unique
from numpy.ma import MaskedArray as ma_maskedarray
from numpy.ma import average as ma_average
from pandas import (
    DataFrame,
    Series,
    concat,
    melt,
    merge,
    read_csv,
    read_excel,
    to_numeric,
)
from PIL import Image
from pyproj import CRS, Transformer
from shapely.geometry import Point
from shapely.wkt import loads
from torch import load as torch_load
from torch import save as torch_save

from process.model import STAGE_INDEX

logger = getLogger()


def read_population(population_path: str):
    """Read population

    Args:
        population_path (str): Population data path
    """
    data = read_excel(population_path, header=6)

    data = data.rename(columns={"Area": "area", "Unnamed: 2": "population"})

    data = data.drop("Unnamed: 1", axis=1)

    # Drop the last row
    data = data.drop(data.index[-1])

    data = data.astype(int)

    data = data[data["area"] > 10000]

    return data


def save_outputs(param_model, workdir):
    if not exists(workdir):
        makedirs(workdir)

    torch_save(param_model["param_model"], join(workdir, "param_model.model"))
    pickle_dump(param_model["params"], open(join(workdir, "params.p"), "wb"))
    pickle_dump(param_model["output_info"], open(join(workdir, "output_info.p"), "wb"))
    pickle_dump(
        param_model["all_interactions"], open(join(workdir, "all_interactions.p"), "wb")
    )
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
            row = sa2[sa2["SA22018_V1_00"] == sa2_code].iloc[
                0
            ]  # Get the corresponding row from y
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
    predict_common_cfg,
    apply_log_for_loss: bool = False,
    plot_err_range: bool = False,
    plot_obs: bool = True,
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
        exposed_counts = count_nonzero(
            output["all_records"] == STAGE_INDEX["exposed"], axis=1
        )
        infected_counts = count_nonzero(
            output["all_records"] == STAGE_INDEX["infected"], axis=1
        )
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

        if vis_cfg["agents_map"]["pop_based_interpolation"]["enable"]:
            pop = read_population(
                vis_cfg["agents_map"]["pop_based_interpolation"]["pop_path"]
            )

        sa2 = read_csv(vis_cfg["agents_map"]["sa2_path"])
        sa2 = sa2.loc[sa2["LAND_AREA_SQ_KM"] > 0]

        sa2 = GeoDataFrame(sa2, geometry=GeoSeries.from_wkt(sa2["WKT"]))
        sa2 = sa2.cx[
            vis_cfg["agents_map"]["domain"]["min_lon"] : vis_cfg["agents_map"][
                "domain"
            ]["max_lon"],
            vis_cfg["agents_map"]["domain"]["min_lat"] : vis_cfg["agents_map"][
                "domain"
            ]["max_lat"],
        ]
        proc_latlons = []
        proc_attrs = []
        proc_indices = []
        for t in range(0, len(outputs[0]["all_target_indices"])):
            if vis_cfg["agents_map"]["individuals"]:
                output = outputs[0]
                proc_index = output["all_target_indices"][t]
                if len(proc_index) == 0:
                    continue

                proc_latlon = obtain_sa2_info(
                    sa2,
                    array(output["agents_area"])[proc_index],
                    output["agents_ethnicity"],
                )
                proc_attr = array(output["agents_ethnicity"])[proc_index]

                scatter_colors = {0: "b", 1: "g", 2: "r", 3: "c", 4: "m"}
                colors = [scatter_colors[val] for val in proc_attr]

                proc_latlons.extend(proc_latlon)
                proc_attrs.extend(colors)

                lats = [lat for lat, _ in proc_latlons]
                lons = [lon for _, lon in proc_latlons]
                # gdf.plot(figsize=(10, 10))
                sa2.plot(figsize=(10, 10))

                scatter(lons, lats, color=proc_attrs, marker="x")
                tight_layout()
                title(f"{t}, Total agents: {len(proc_latlons)}")
                savefig(join(tmp_dir, f"agents_{t}.png"), bbox_inches="tight")
                close()
            else:
                print(f"    agent map at {t}")
                for output in outputs:
                    proc_index = output["all_target_indices"][t]
                    if len(proc_index) == 0:
                        continue

                    proc_indices.extend(proc_index)

                all_agents_areas = array(output["agents_area"])[proc_indices]
                occurrences = bincount(all_agents_areas)

                occurrences_dict = dict(
                    zip(unique(all_agents_areas), occurrences[unique(all_agents_areas)])
                )
                sa2["agents_occurrences"] = sa2["SA22018_V1_00"].map(occurrences_dict)
                sa2["agents_occurrences"] = sa2["agents_occurrences"].fillna(0.0)
                # sa2["agents_occurrences"] = sa2["agents_occurrences"] / len(outputs)

                mask = (sa2["agents_occurrences"] > 0) & (
                    sa2["agents_occurrences"] < 1.0
                )
                sa2.loc[mask, "agents_occurrences"] = 1.0

                sa2.loc[sa2["agents_occurrences"] == 0.0, "agents_occurrences"] = NaN

                if vis_cfg["agents_map"]["pop_based_interpolation"]["enable"]:
                    sa2["representative_point"] = sa2.representative_point()
                    if "population" not in sa2.columns:
                        sa2 = sa2.merge(
                            pop, left_on="SA22018_V1_00", right_on="area", how="left"
                        )

                    for index, row in sa2[
                        sa2["agents_occurrences"].isnull()
                    ].iterrows():
                        target_geometry = row["representative_point"]

                        # Find the nearest geometry and index
                        nearest_index = list(
                            sa2["representative_point"]
                            .distance(target_geometry)
                            .nsmallest(3)
                            .index
                        )
                        nearest_index.remove(index)
                        nearest_geometry = sa2.loc[
                            nearest_index,
                            [
                                "SA22018_V1_00",
                                "agents_occurrences",
                                "geometry",
                                "population",
                            ],
                        ]

                        # if not nearest_geometry["agents_occurrences"].isna().all():
                        #    x = 3

                        ma = ma_maskedarray(
                            nearest_geometry["agents_occurrences"],
                            mask=isnan(
                                nearest_geometry["agents_occurrences"],
                            ),
                        )
                        weighted_avg = ma_average(
                            ma, weights=nearest_geometry["population"]
                        )

                        sa2.at[index, "agents_occurrences"] = weighted_avg

                _, ax = subplots(1, 1, figsize=(10, 10))
                sa2.plot(
                    column="agents_occurrences",
                    cmap="Reds",
                    legend=True,
                    ax=ax,
                    legend_kwds={"shrink": 0.3},
                    vmin=0,
                    vmax=int(output["pred"].sum().item() * 0.05),
                )
                sa2.boundary.plot(linewidth=0.3, color="k", linestyle="solid", ax=ax)
                ax.set_xlim(
                    vis_cfg["agents_map"]["domain"]["min_lon"],
                    vis_cfg["agents_map"]["domain"]["max_lon"],
                )
                ax.set_ylim(
                    vis_cfg["agents_map"]["domain"]["min_lat"],
                    vis_cfg["agents_map"]["domain"]["max_lat"],
                )

                if vis_cfg["agents_map"]["highlight_sa2"] is not None:
                    highlighted_areas = sa2[
                        sa2["SA22018_V1_00"].isin(
                            vis_cfg["agents_map"]["highlight_sa2"]
                        )
                    ]
                    highlighted_areas.plot(
                        ax=ax,
                        color="blue",  # You can choose any color you want
                        legend=False,  # You can adjust this based on your needs
                        # label="Highlighted Areas",  # Label for the legend
                    )

                tight_layout()
                # title(f"{t}, {total_agents}")
                savefig(join(tmp_dir, f"agents_{t}.png"), bbox_inches="tight")
                close()

        png_files = [f for f in listdir(tmp_dir) if f.endswith(".png")]
        png_files = sorted(
            png_files, key=lambda item: int(item.split("_")[1].split(".")[0])
        )
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
        if i == 0 and plot_obs:
            my_targ = output["y"].tolist()
            plot(my_targ, color="k", linewidth=2.0, label="Observed cases")

        if len(outputs) == 1:
            plot(my_pred, linewidth=1.0, linestyle="--")
        else:
            all_preds.append(my_pred)

    x = range(len(all_preds[0]))
    if plot_err_range:
        all_preds = array(all_preds)
        y_min = min(all_preds, 0)
        y_max = max(all_preds, 0)
        ax.fill_between(x, y_min, y_max, alpha=0.5)
    else:
        for proc_prd in all_preds:
            plot(proc_prd, linewidth=0.75, color="grey", alpha=0.5, linestyle="--")

    start_name = 0
    if predict_common_cfg["start"]["name"] is not None:
        start_name = predict_common_cfg["start"]["name"]

    end_time = start_name + len(x)
    if predict_common_cfg["end"]["name"] is not None:
        end_time = predict_common_cfg["end"]["name"]

    #    # Set the x-axis tick positions and labels
    xtick_positions = list(range(start_name, end_time))
    tick_labels = [str(tick) for tick in xtick_positions]
    ax.set_xticks(x[::3])
    ax.set_xticklabels(tick_labels[::3])

    legend()
    if vis_cfg["pred"]["title_str"] is not None:
        title(vis_cfg["pred"]["title_str"])
    else:
        title(
            f"JUNE-NZ validation {vis_cfg['name']} \n Simulation: {int(sum(my_pred))} vs Observed: {int(sum(my_targ))}"
        )
    xlabel(f"{vis_cfg['temporal_res']}")
    ylabel("Number of cases")
    tight_layout()
    savefig(join(workdir, "prediction_vs_truth.png"), bbox_inches="tight")
    close()
