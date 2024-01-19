from glob import glob
from logging import getLogger
from os.path import join
from pickle import dump as pickle_dump
from pickle import load as pickle_load
from random import choice as random_choice
from re import findall as re_findall

from numpy.random import random as numpy_random
from pandas import DataFrame, concat
from pandas import merge as pandas_merge
from pandas import read_csv as pandas_read_csv
from pandas import read_parquet

from process.input import (
    AGE_INDEX,
    ETHNICITY_INDEX,
    GENDER_INDEX,
    LOC_INDEX,
    TRAINING_ENS_MEMBERS,
)
from process.input.vis import agents_vis

logger = getLogger()


def get_diary_data(syn_data_path: str, diary_data_path: str) -> DataFrame:
    """Create agent and diary data

    Args:
        syn_data_path (str): _description_
        diary_data_path (str): _description_

    Returns:
        DataFrame: diary data
    """
    agents = pandas_read_csv(syn_data_path)
    diary_data = pickle_load(open(diary_data_path, "rb"))["diaries"]

    # diary_data = diary_data[[12, "id"]]
    df_melted = diary_data.melt(id_vars="id", var_name="hour", value_name="spec")
    merged_df = pandas_merge(df_melted, agents, on="id", how="left")

    for proc_key in [
        "household",
        "supermarket",
        "restaurant",
        "travel",
        "school",
        "company",
    ]:
        proc_key_to_map = proc_key
        if proc_key == "travel":
            proc_key_to_map = "public_transport_trip"

        merged_df.loc[merged_df["spec"] == proc_key, "group"] = merged_df[
            proc_key_to_map
        ]

    merged_df["group"] = merged_df["group"].apply(
        lambda x: random_choice(x.split(",")) if "," in str(x) else x
    )

    # there might be private travel which does not have a group value
    x = 3
    return {
        "agents": agents,
        "interaction": merged_df[["id", "group", "spec", "area"]].dropna(),
    }


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
    last_row_df_transposed.to_csv(
        join(workdir, "output.csv"), index=False, header=False
    )


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


def _agent_and_interaction_preproc(
    data: dict, vaccine_ratio_cfg: dict, sa2: list
) -> dict:
    """Obtain agent and interaction data:

    Args:
        data (dict): Input data
        vaccine_ratio_cfg (dict): Vaccine ratio
        sa2 (list): SA2 to be processed

    Returns:
        dict: processed agent and interaction
    """

    def _assign_vaccination(ethnicity: str):
        """Assign vaccination status

        Args:
            ethnicity (str): ethnicity such as Maori, European, Asian etc.

        Returns:
            _type_: _description_
        """
        return "yes" if numpy_random() < vaccine_ratio_cfg[ethnicity] else "no"

    interaction_data = data["interaction"]
    agents_data = data["agents"]
    agents_data["vaccine"] = agents_data["ethnicity"].apply(_assign_vaccination)

    if sa2 is not None:
        interaction_data = interaction_data[interaction_data["area"].isin(sa2)]
        agents_data = agents_data[agents_data["area"].isin(sa2)]

    agents_data = agents_data.reset_index()
    agents_data["row_number"] = agents_data.index

    interaction_data = interaction_data[["id", "group", "spec"]]
    interaction_data = pandas_merge(
        interaction_data, agents_data[["id", "row_number"]], on="id", how="left"
    )
    interaction_data["id"] = interaction_data["row_number"]
    interaction_data = interaction_data.drop("row_number", axis=1)

    agents_data["id"] = agents_data["row_number"]
    agents_data = agents_data.drop(["index", "row_number"], axis=1)

    return {"agents": agents_data, "interaction": interaction_data}


def write_agent_and_interactions(
    data: dict,
    sa2: list,
    data_dir: str,
    interaction_ratio_cfg: dict,
    vaccine_ratio_cfg: dict,
    max_interaction_for_each_venue: int or None = None,
):
    """Write out agents and agent interactions

    Args:
        data (dict): _description_
        sa2 (list): _description_
        data_dir (str): _description_
        interaction_ratio_cfg (dict): _description_
        vaccine_ratio_cfg (dict): _description_
        max_interaction_for_each_venue (intorNone, optional): _description_. Defaults to None.
    """
    data = _agent_and_interaction_preproc(data, vaccine_ratio_cfg, sa2)

    interaction_data = data["interaction"]
    agents_data = data["agents"]

    id_num = len(interaction_data["id"].unique())

    logger.info(f"Total population: {id_num}")

    for proc_member in range(TRAINING_ENS_MEMBERS):
        logger.info(f"Processing member {proc_member} / {TRAINING_ENS_MEMBERS}")
        sampled_df = []
        for spec_value in list(interaction_data["spec"].unique()):
            logger.info(
                f"   Processing {spec_value}: {interaction_ratio_cfg[spec_value]} ..."
            )
            sampled_df.append(
                select_rows_by_spec(
                    interaction_data, spec_value, interaction_ratio_cfg[spec_value]
                )
            )

        logger.info(f"   Combining all interaction inputs ...")
        sampled_df = concat(sampled_df, ignore_index=True).drop_duplicates()

        if max_interaction_for_each_venue is not None:
            logger.info(
                f"   Limiting interactions to {max_interaction_for_each_venue} ..."
            )
            sampled_df = sampled_df.groupby("group", group_keys=False).apply(
                lambda x: x.sample(
                    min(len(x), max_interaction_for_each_venue), random_state=1
                )
            )

        logger.info("   Start interaction merging process ...")
        interactions = pandas_merge(sampled_df, sampled_df, on="group")

        logger.info("   Filtering out self interactions ...")
        interactions = interactions[interactions["id_x"] != interactions["id_y"]]

        logger.info("   Selecting subset of columns ...")
        interactions = interactions[["id_x", "id_y", "spec_x", "group"]]

        logger.info("   Removing hospitals ...")
        interactions = interactions[interactions["spec_x"] != "hospital"]

        logger.info("   Mapping location index ...")
        interactions["spec"] = interactions["spec_x"].map(LOC_INDEX)

        interactions = interactions[["id_x", "id_y", "group", "spec"]].drop_duplicates()

        logger.info(interactions)

        logger.info(f"   Total interactions: {len(interactions)} ...")

        for proc_loc_type in LOC_INDEX:
            proc_occurrences = len(
                interactions[interactions["spec"] == LOC_INDEX[proc_loc_type]]
            )
            logger.info(f"    - {proc_loc_type}: {proc_occurrences}")

        logger.info(f"   Writing outputs ...")
        interactions.to_parquet(
            join(data_dir, f"interaction_graph_cfg_member_{proc_member}.parquet")
        )

    # write out agents
    agents_data.to_parquet(join(data_dir, f"agents.parquet"))
