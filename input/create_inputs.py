from os.path import join

from numpy.random import random as numpy_random
from pandas import DataFrame
from pandas import merge as pandas_merge
from pandas import read_parquet

from input import AGE_INDEX, ETHNICITY_INDEX, LOC_INDEX, SEX_INDEX


def read_june_nz_inputs(june_nz_data) -> DataFrame:
    data = read_parquet(june_nz_data)
    return data.drop_duplicates()


def convert_to_age_index(age):
    for age_range, index in AGE_INDEX.items():
        age_start, age_end = map(int, age_range.split("-"))
        if age >= age_start and age <= age_end:
            return index


def get_agents(data, data_dir, vaccine_ratio):
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
    agents.to_parquet(join(data_dir, "agents.parquet"), index=False)


def get_interactions(data, data_dir):
    data = data[["id", "group", "spec"]]
    interactions = pandas_merge(data, data, on="group")

    # Filter out self-interactions
    interactions = interactions[interactions["id_x"] != interactions["id_y"]]

    interactions = interactions[["id_x", "id_y", "spec_x"]]

    interactions = interactions[interactions["spec_x"] != "hospital"]

    interactions["spec_x"] = interactions["spec_x"].map(LOC_INDEX)

    interactions = interactions.drop_duplicates()

    interactions.to_parquet(join(data_dir, "interaction_graph_cfg.parquet"))
