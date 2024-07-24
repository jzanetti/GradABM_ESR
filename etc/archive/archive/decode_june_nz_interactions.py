import random
from os import makedirs
from os.path import exists, join

import matplotlib.pyplot as plt
import numba as nb
from pandas import DataFrame, merge, read_parquet

loc_index = {
    "household": 0,
    "city_transport": 1,
    "inter_city_transport": 2,
    "gym": 3,
    "grocery": 4,
    "pub": 5,
    "cinema": 6,
    "school": 7,
    "company": 8,
}

age_index = {"0-10": 0, "11-20": 1, "21-30": 2, "31-40": 3, "41-50": 4, "51-60": 5, "61-999": 6}

ethnicity_index = {"European": 0, "Maori": 1, "Pacific": 2, "Asian": 3, "MELAA": 4}

sex_index = {"m": 0, "f": 1}

# vaccine_ratio = {"European": 0.8, "Maori": 0.6, "Pacific": 0.6, "Asian": 0.8, "MELAA": 0.8}
vaccine_ratio = {"European": 0.9, "Maori": 0.7, "Pacific": 0.7, "Asian": 0.9, "MELAA": 0.9}


def convert_to_age_index(age):
    for age_range, index in age_index.items():
        age_start, age_end = map(int, age_range.split("-"))
        if age >= age_start and age <= age_end:
            return index


def get_agents(data, data_dir):
    def _assign_vaccination(ethnicity):
        return 1 if random.random() < vaccine_ratio[ethnicity] else 0

    # agents = data[["id", "age"]].drop_duplicates()
    # count_with_count = (x["vaccine_coverage"] == 1.0).sum()
    agents = data[["id", "age", "sex", "ethnicity", "area"]].drop_duplicates()
    print(f"Total population {len(agents)}")
    agents["age"] = agents["age"].apply(convert_to_age_index)
    agents["sex"] = agents["sex"].map(sex_index)
    agents["vaccine"] = agents["ethnicity"].apply(_assign_vaccination)
    agents["ethnicity"] = agents["ethnicity"].map(ethnicity_index)
    agents.to_parquet(join(data_dir, "agents.parquet"), index=False)


def get_interactions(data, data_dir):
    data = data[["id", "group", "spec"]]
    interactions = merge(data, data, on="group")

    # Filter out self-interactions
    interactions = interactions[interactions["id_x"] != interactions["id_y"]]

    interactions = interactions[["id_x", "id_y", "spec_x"]]

    interactions = interactions[interactions["spec_x"] != "hospital"]

    interactions["spec_x"] = interactions["spec_x"].map(loc_index)

    interactions = interactions.drop_duplicates()

    interactions.to_parquet(join(data_dir, "interaction_graph_cfg.parquet"))


def interaction_groups(data):
    for proc_spec in list(data["spec"].unique()):
        proc_data = data[data["spec"] == proc_spec]
        counts_df = proc_data.groupby("group")["id"].nunique().reset_index()
        plt.plot(counts_df["id"])
        plt.title(f"Mean number of agents: {round(counts_df['id'].mean(), 3)}")
        plt.xlabel(proc_spec)
        plt.ylabel("Number of agents")
        plt.savefig(f"interaction_diags_{proc_spec}.png")
        plt.close()


data_dir = "data/exp4"

if not exists(data_dir):
    makedirs(data_dir)

data = read_parquet("interaction_output.parquet")
data = data.drop_duplicates()

create_diags = False
if create_diags:
    interaction_groups(data)

# get_interactions2(data[0:1000], data_dir)
print("Creating agents ...")
get_agents(data, data_dir)

print("Creating interactions ...")
get_interactions(data, data_dir)

print("done")
