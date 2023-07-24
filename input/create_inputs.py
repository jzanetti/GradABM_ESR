from logging import getLogger
from os.path import join

from numpy.random import random as numpy_random
from pandas import DataFrame
from pandas import merge as pandas_merge
from pandas import read_parquet

from input import (
    AGE_INDEX,
    ETHNICITY_INDEX,
    INTERACTION_ENS_MEMBERS,
    LOC_INDEX,
    SEX_INDEX,
)

logger = getLogger()


def write_target(workdir: str, target_path: str or None, dhb_list: list):
    if target_path is None:
        return
    target = read_parquet(target_path)

    if dhb_list is not None:
        target = target[target["Region"].isin(dhb_list)]

    target.fillna(0, inplace=True)

    target = target.set_index("Region")
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
    data = read_parquet(june_nz_data)
    return data.drop_duplicates()


def convert_to_age_index(age):
    for age_range, index in AGE_INDEX.items():
        age_start, age_end = map(int, age_range.split("-"))
        if age >= age_start and age <= age_end:
            return index


def get_agents(data, sa2, data_dir, vaccine_ratio):
    def _assign_vaccination(ethnicity):
        return 1 if numpy_random() < vaccine_ratio[ethnicity] else 0

    # agents = data[["id", "age"]].drop_duplicates()
    # count_with_count = (x["vaccine_coverage"] == 1.0).sum()
    agents = data[["id", "age", "sex", "ethnicity", "area"]].drop_duplicates()
    print(f"Total population {len(agents)}")
    agents["age"] = agents["age"].apply(convert_to_age_index)
    agents["sex"] = agents["sex"].map(SEX_INDEX)
    agents["vaccine"] = agents["ethnicity"].apply(_assign_vaccination)
    agents["ethnicity"] = agents["ethnicity"].map(ETHNICITY_INDEX)

    # if sa2 is not None:
    #    agents = agents[agents["area"].isin(sa2)]

    # matching the number of rows with "id":
    # unique_ids = agents["id"].unique()
    # max_id = agents["id"].max()
    # all_ids = range(1, max_id + 1)
    # agents = agents.set_index("id").reindex(all_ids, fill_value=-999).reset_index()

    agents.to_parquet(join(data_dir, "agents.parquet"), index=False)

    agents["row_number"] = range(len(agents))

    return agents


def get_interactions(data, agents, sa2, data_dir, percentage_each_member: float = 0.5):
    if sa2 is not None:
        data = data[data["area"].isin(sa2)]

    data = data[["id", "group", "spec"]]

    for proc_member in range(INTERACTION_ENS_MEMBERS):
        sampled_df = data.sample(frac=percentage_each_member)

        logger.info(f"Processing member {proc_member} / {INTERACTION_ENS_MEMBERS}")

        logger.info("   Start interaction merging process ...")
        interactions = pandas_merge(sampled_df, sampled_df, on="group")

        logger.info("   Filtering out self interactions ...")
        interactions = interactions[interactions["id_x"] != interactions["id_y"]]

        interactions = interactions[["id_x", "id_y", "spec_x"]]

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
