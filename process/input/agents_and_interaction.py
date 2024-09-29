from logging import getLogger
from os.path import join
from random import choice as random_choice

from numpy import concatenate as numpy_concatenate
from numpy.random import choice as numpy_choice
from numpy.random import random as numpy_random
from pandas import DataFrame, concat
from pandas import merge as pandas_merge
from pandas import read_parquet as pandas_read_parquet

from process import LOC_INDEX

logger = getLogger()


def get_diary_data(syn_data_path: str, diary_data_path: str) -> DataFrame:
    """Create agent and diary data

    Args:
        syn_data_path (str): _description_
        diary_data_path (str): _description_

    Returns:
        DataFrame: diary data
    """
    agents = pandas_read_parquet(syn_data_path)
    diary_data = pandas_read_parquet(diary_data_path)

    df_melted = diary_data.melt(id_vars="id", var_name="hour", value_name="spec")
    merged_df = pandas_merge(df_melted, agents, on="id", how="left")

    """
    return {
        "agents": agents,
        "interaction": merged_df[["id", "spec", "area"]].dropna(),
    }
    """

    for proc_key in list(merged_df["spec"].unique()):
        proc_key_to_map = proc_key
        if proc_key == "travel":
            proc_key_to_map = "public_transport_trip"
        #if proc_key == "restaurant":
        #    proc_key_to_map = "restauraunt"

        merged_df.loc[merged_df["spec"] == proc_key, "group"] = merged_df[
            proc_key_to_map
        ]

    merged_df["group"] = merged_df["group"].apply(
        lambda x: random_choice(x.split(",")) if "," in str(x) else x
    )

    # there might be private travel which does not have a group value
    return {
        "agents": agents,
        "interaction": merged_df[["id", "group", "spec", "area"]].dropna(),
    }


def write_target(
    workdir: str,
    target_path: str or None,  # type: ignore
    dhb_list: list,
    target_index_range: dict or None,  # type: ignore
):
    if target_path is None:
        return
    target = pandas_read_parquet(target_path)
    target.fillna(0, inplace=True)
    target = target.set_index("Region")

    # Create the list of all weeks using a for loop
    all_weeks = ["Week_" + str(week) for week in range(0, 54)]
    target = target.reindex(columns=all_weeks, fill_value=0)

    if dhb_list is not None:
        # target = target[target["Region"].isin(dhb_list)]
        target = target.loc[dhb_list]

    if target_index_range is not None:
        target = target[
            [
                f"Week_{i}"
                for i in range(target_index_range["start"], target_index_range["end"])
            ]
        ]

    target = target.astype(float)
    target.loc["Total"] = target.sum(axis=0)

    # Get the last row of the DataFrame as a separate DataFrame
    last_row_df = target.tail(1)
    last_row_df_transposed = last_row_df.T
    last_row_df_transposed.columns = ["target"]

    # Write the last row DataFrame to a CSV file
    last_row_df_transposed.to_parquet(join(workdir, "target.parquet"), index=False)


def get_sa2_from_dhb(dhb_sa2_map_data_path, dhb_list):
    if dhb_sa2_map_data_path is None:
        return None

    if dhb_list is None:
        return None

    sa2_to_dhb = pandas_read_parquet(dhb_sa2_map_data_path)

    sa2_to_dhb = sa2_to_dhb[sa2_to_dhb["DHB_name"].isin(dhb_list)]

    return list(sa2_to_dhb["SA2"].unique())


def select_rows_by_spec(df, spec_value, percentage):
    spec_rows = df[df["spec"] == spec_value]
    selected_rows = spec_rows.sample(frac=percentage)
    return selected_rows


def _agent_and_interaction_preproc(data: dict, runtime_cfg: dict, sa2: list) -> dict:
    """Obtain agent and interaction data:

    Args:
        data (dict): Input data
        runtime_cfg (dict): Runtime cfg such as Vaccine ratio
        sa2 (list): SA2 to be processed

    Returns:
        dict: processed agent and interaction
    """

    def _assign_vaccine(row):
        """Assign vaccination status

        Args:
            ethnicity (str): ethnicity such as Maori, European, Asian etc.

        Returns:
            _type_: _description_
        """

        def _get_age_range(age):
            if age <= 5:
                return "0-5"
            elif 6 <= age <= 50:
                return "6-50"
            else:
                return "50-999"

        age_range = _get_age_range(row["age"])
        ethnicity = row["ethnicity"]
        proc_coverage = runtime_cfg["vaccine"][ethnicity]

        if age_range not in proc_coverage:
            raise Exception(
                f"Not able to find the proper age range for vaccination: {age_range}"
            )
        coverage = proc_coverage[age_range]  # Default to 1.0 if not found
        return numpy_choice(["yes", "no"], p=[coverage, 1 - coverage])

        # return "yes" if numpy_random() < runtime_cfg["vaccine"][ethnicity] else "no"

    interaction_data = data["interaction"]
    agents_data = data["agents"]

    if "vaccine" in runtime_cfg:
        agents_data["vaccine"] = agents_data[["age", "ethnicity"]].apply(
            _assign_vaccine, axis=1
        )
        # agents_data["vaccine"] = agents_data["ethnicity"].apply(_assign_vaccine)

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


def set_max_interaction_for_each_group(
    sampled_df: DataFrame, max_interaction_for_each_venue: int or None
) -> DataFrame:
    """Make each interaction for a group less than a threshold

    Args:
        sampled_df (DataFrame): dataframe to be processed

    Returns:
        DataFrame: processed dataframe
    """

    if max_interaction_for_each_venue is None:
        return sampled_df

    sampled_df_counts = sampled_df["group"].value_counts()
    over_max_group = sampled_df_counts[
        sampled_df_counts > max_interaction_for_each_venue
    ].index

    over_max_index = numpy_concatenate(
        [
            numpy_choice(
                sampled_df[sampled_df["group"] == group].index,
                max_interaction_for_each_venue,
                replace=False,
            )
            for group in over_max_group
        ]
    )
    under_max_index = sampled_df[~sampled_df["group"].isin(over_max_group)].index
    return sampled_df.loc[numpy_concatenate([over_max_index, under_max_index])]


def create_agents_and_interactions(
    data: dict,
    sa2: list,
    interaction_ratio_cfg: dict,
    runtime_cfg: dict,
    max_interaction_for_each_venue: int or None = None,  # type: ignore
):
    """Write out agents and agent interactions

    Args:
        data (dict): _description_
        sa2 (list): _description_
        data_dir (str): _description_
        interaction_ratio_cfg (dict): _description_
        runtime_cfg (dict): runtime configuration such as vaccine ratio
        max_interaction_for_each_venue (int or None, optional): _description_. Defaults to None.
    """

    data = _agent_and_interaction_preproc(data, runtime_cfg, sa2)

    interaction_data = data["interaction"]

    sampled_df = []
    for spec_value in list(interaction_data["spec"].unique()):
        logger.info(
            f"    * Processing {spec_value}: {interaction_ratio_cfg[spec_value] * 100}% ..."
        )
        sampled_df.append(
            select_rows_by_spec(
                interaction_data, spec_value, interaction_ratio_cfg[spec_value]
            )
        )

    logger.info(f"   Combining all interaction inputs ...")
    sampled_df = concat(sampled_df, ignore_index=True).drop_duplicates()

    logger.info(f"   Setting max interaction for each group ...")
    sampled_df = set_max_interaction_for_each_group(
        sampled_df, max_interaction_for_each_venue
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

    logger.info(f"   Total interactions: {len(interactions)} ...")

    for proc_loc_type in LOC_INDEX:
        proc_occurrences = len(
            interactions[interactions["spec"] == LOC_INDEX[proc_loc_type]]
        )
        logger.info(f"    * {proc_loc_type}: {proc_occurrences}")

    return {"agents": data["agents"], "interactions": interactions}
