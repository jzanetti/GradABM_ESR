# https://medium.com/mlearning-ai/ultimate-guide-to-graph-neural-networks-1-cora-dataset-37338c04fe6f
# https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html#epidemiological-forecasting

import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy
import pandas
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

EGDE_COLOR_INDEX = {
    0: ["household", "k"],
    1: ["supermarket", "k"],
    2: ["company", "r"],
    3: ["school", "k"],
    4: ["travel", "k"],
    5: ["restaurant", "k"],
    6: ["pharmacy", "k"],
}

# 136000: Eden Terrace, 133200: Queen Street
agents_datapath = "/tmp/gradabm_esr/Auckland_2019_measles3/input/agents.parquet"
interactions_datapath = [
    "/tmp/gradabm_esr/Auckland_2019_measles3/input/interaction_graph_cfg_member_0_0.parquet",
    "/tmp/gradabm_esr/Auckland_2019_measles3/input/interaction_graph_cfg_member_0_1.parquet",
    "/tmp/gradabm_esr/Auckland_2019_measles3/input/interaction_graph_cfg_member_0_2.parquet",
    "/tmp/gradabm_esr/Auckland_2019_measles3/input/interaction_graph_cfg_member_1_0.parquet",
    "/tmp/gradabm_esr/Auckland_2019_measles3/input/interaction_graph_cfg_member_1_1.parquet",
    "/tmp/gradabm_esr/Auckland_2019_measles3/input/interaction_graph_cfg_member_1_2.parquet",
]
address_datapath = "/tmp/syspop_test/Auckland/syspop_location.parquet"


# pos = {agent_id: (lon, lat) for agent_id, lon, lat in zip(agents_data['id'], agents_data['lon'], agents_data['lat'])}
agents_data = pandas.read_parquet(agents_datapath)
interactions_data = [
    pandas.read_parquet(file_path) for file_path in interactions_datapath
]
address_data = pandas.read_parquet(address_datapath)

# Combine the DataFrames vertically
interactions_data = pandas.concat(interactions_data, ignore_index=True)
interactions_data = interactions_data[interactions_data["spec"].isin([3, 4])]

interactions_data = interactions_data.sample(5000)

old_node_ids = list(interactions_data["id_x"].unique()) + list(
    interactions_data["id_y"].unique()
)

new_id = {}
for i, proc_id in enumerate(old_node_ids):
    new_id[proc_id] = i

interactions_data["id_x"] = interactions_data["id_x"].map(new_id)
interactions_data["id_y"] = interactions_data["id_y"].map(new_id)

merged_df = pandas.merge(
    interactions_data, address_data, left_on="group", right_on="name"
)
merged_df = merged_df.drop_duplicates()
node_coordinates = {}
for _, row in merged_df.iterrows():
    node_id = row.id_x
    latitude = row.latitude + random.uniform(-0.0001, 0.0001)
    longitude = row.longitude + random.uniform(-0.0001, 0.0001)
    node_coordinates[node_id] = {"lat": latitude, "lon": longitude}


for _, row in merged_df.iterrows():
    node_id = row.id_y
    if node_id in node_coordinates:
        continue
    latitude = row.latitude + random.uniform(-0.0001, 0.0001)
    longitude = row.longitude + random.uniform(-0.0001, 0.0001)
    node_coordinates[node_id] = {"lat": latitude, "lon": longitude}

idx = []
idy = []
for i, row in merged_df.iterrows():
    idx.append(row["id_x"])
    idy.append(row["id_y"])

edge_index = torch.tensor(numpy.array([idx, idy]))
all_nodes = edge_index.flatten()

# Create pos tensor from node coordinates
pos = torch.tensor(
    [
        [node_coordinates[int(node)]["lat"], node_coordinates[int(node)]["lon"]]
        for node in all_nodes
    ],
    dtype=torch.float,
)

print("xxx")
data = Data(edge_index=torch.tensor(numpy.array([idx, idy])), pos=pos)
G = to_networkx(data, to_undirected=False)
nx.draw_networkx(G, pos, width=0.3, alpha=0.5, with_labels=False, node_size=5)

plt.savefig("test.png")
plt.close()
