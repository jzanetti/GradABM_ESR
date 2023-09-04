from glob import glob
from logging import getLogger
from os.path import join
from pickle import dump as pickle_dump
from re import findall as re_findall

from numpy.random import random as numpy_random
from pandas import DataFrame, concat
from pandas import merge as pandas_merge
from pandas import read_parquet

from input import AGE_INDEX, ETHNICITY_INDEX, LOC_INDEX, SEX_INDEX, TRAINING_ENS_MEMBERS
from input.vis import agents_vis

logger = getLogger()


def write_target(workdir: str, target_path: str or None, dhb_list: list):
    if target_path is None:
        return
    target = read_parquet(target_path)
    target.fillna(0, inplace=True)
    target = target.set_index("Region")

    # Create the list of all weeks using a for loop
    all_weeks = ["Week_" + str(week) for week in range(0, 54)]
    target = target.reindex(columns=all_weeks, fill_value=0)

    if dhb_list is not None:
        # target = target[target["Region"].isin(dhb_list)]
        target = target.loc[dhb_list]

    target = target.astype(float)
    target.loc["Total"] = target.sum(axis=0)

    # Get the last row of the DataFrame as a separate DataFrame
    last_row_df = target.tail(1)
    last_row_df_transposed = last_row_df.T
    last_row_df_transposed.columns = ["target"]

    # Write the last row DataFrame to a CSV file
    last_row_df_transposed.to_csv(join(workdir, "output.csv"), index=False, header=False)


def get_sa2_from_dhb(dhb_sa2_map_data_path, dhb_list):
    if dhb_sa2_map_data_path is None:
        return None

    if dhb_list is None:
        return None

    sa2_to_dhb = read_parquet(dhb_sa2_map_data_path)

    sa2_to_dhb = sa2_to_dhb[sa2_to_dhb["DHB_name"].isin(dhb_list)]

    return list(sa2_to_dhb["SA2"].unique())


def read_june_nz_inputs(june_nz_data) -> DataFrame:
    if june_nz_data.endswith("parquet"):
        data = read_parquet(june_nz_data)
    else:
        data = []
        for proc_file in glob(june_nz_data + "/*.parquet"):
            proc_data = read_parquet(proc_file)
            proc_time = re_findall(r"\d{8}T\d{2}", proc_file)[0]
            proc_data["time"] = proc_time
            data.append(proc_data)

        data = concat(data, ignore_index=True)
    data = data.drop_duplicates()

    return data


def convert_to_age_index(age):
    for age_range, index in AGE_INDEX.items():
        age_start, age_end = map(int, age_range.split("-"))
        if age >= age_start and age <= age_end:
            return index


def get_agents(data, sa2, data_dir, vaccine_ratio, plot_agents=False):
    def _assign_vaccination(ethnicity):
        return 1 if numpy_random() < vaccine_ratio[ethnicity] else 0

    if sa2 is not None:
        data = data[data["area"].isin(sa2)]

    agents = data[["id", "age", "sex", "ethnicity", "area"]].drop_duplicates()
    print(f"Total population {len(agents)}")
    agents["age"] = agents["age"].apply(convert_to_age_index)
    agents["sex"] = agents["sex"].map(SEX_INDEX)
    agents["vaccine"] = agents["ethnicity"].apply(_assign_vaccination)
    agents["ethnicity"] = agents["ethnicity"].map(ETHNICITY_INDEX)

    agents.to_parquet(join(data_dir, "agents.parquet"), index=False)
    pickle_dump(sa2, open(join(data_dir, "all_areas.p"), "wb"))

    agents["row_number"] = range(len(agents))

    if plot_agents:
        agents_vis(agents, sa2)

    agents_group = data[["id", "group"]].drop_duplicates()
    agents_group.to_parquet(join(data_dir, "agents_group.parquet"), index=False)

    return agents


def select_rows_by_spec(df, spec_value, percentage):
    spec_rows = df[df["spec"] == spec_value]
    selected_rows = spec_rows.sample(frac=percentage)
    return selected_rows


def get_interactions(data, agents, sa2, data_dir, interaction_ratio_cfg):
    if sa2 is not None:
        data = data[data["area"].isin(sa2)]

    data = data[["id", "group", "spec"]]

    id_num = len(data["id"].unique())

    logger.info(f"Total population: {id_num}")

    for proc_member in range(TRAINING_ENS_MEMBERS):
        logger.info(f"Processing member {proc_member} / {TRAINING_ENS_MEMBERS}")
        sampled_df = []
        for spec_value in list(data["spec"].unique()):
            logger.info(f"   Processing {spec_value}: {interaction_ratio_cfg[spec_value]} ...")
            sampled_df.append(
                select_rows_by_spec(data, spec_value, interaction_ratio_cfg[spec_value])
            )

        logger.info(f"   Combining all interaction inputs ...")
        sampled_df = concat(sampled_df, ignore_index=True)

        logger.info("   Start interaction merging process ...")
        interactions = pandas_merge(sampled_df, sampled_df, on="group")

        logger.info("   Filtering out self interactions ...")
        interactions = interactions[interactions["id_x"] != interactions["id_y"]]
        # mask = interactions["id_x"] != interactions["id_y"]
        # interactions = interactions[mask]

        logger.info("   Selecting subset of columns ...")
        interactions = interactions[["id_x", "id_y", "spec_x"]]

        logger.info("   Removing hospitals ...")
        interactions = interactions[interactions["spec_x"] != "hospital"]

        logger.info("   Mapping location index ...")
        interactions["spec_x"] = interactions["spec_x"].map(LOC_INDEX)

        interactions = interactions.drop_duplicates()

        for proc_id_name in ["id_x", "id_y"]:
            merged_df = interactions.merge(agents, left_on=proc_id_name, right_on="id")
            if proc_id_name == "id_x":
                columns_to_keep = ["id_y", "spec_x", "row_number"]
            else:
                columns_to_keep = ["id_x", "spec_x", "row_number"]

            interactions = merged_df[columns_to_keep]
            interactions = interactions.rename(columns={"row_number": proc_id_name})

        interactions = interactions.reindex(columns=["id_x", "id_y", "spec_x"])

        logger.info(f"   Total interactions: {len(interactions)} ...")

        logger.info(f"   Writing outputs ...")
        interactions.to_parquet(
            join(data_dir, f"interaction_graph_cfg_member_{proc_member}.parquet")
        )
