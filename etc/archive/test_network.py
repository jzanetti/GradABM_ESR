# https://medium.com/mlearning-ai/ultimate-guide-to-graph-neural-networks-1-cora-dataset-37338c04fe6f
# https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html#epidemiological-forecasting

import random

import matplotlib.pyplot as plt
import networkx as nx
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

sample_size = 1000

agents_datapath = "/tmp/gradabm_esr/Auckland_2019_measles/input/agents.parquet"
interactions_datapath = [
    "/tmp/gradabm_esr/Auckland_2019_measles/input/interaction_graph_cfg_member_0_0.parquet",
    "/tmp/gradabm_esr/Auckland_2019_measles/input/interaction_graph_cfg_member_0_1.parquet",
    "/tmp/gradabm_esr/Auckland_2019_measles/input/interaction_graph_cfg_member_0_2.parquet",
    "/tmp/gradabm_esr/Auckland_2019_measles/input/interaction_graph_cfg_member_1_0.parquet",
    "/tmp/gradabm_esr/Auckland_2019_measles/input/interaction_graph_cfg_member_1_1.parquet",
    "/tmp/gradabm_esr/Auckland_2019_measles/input/interaction_graph_cfg_member_1_2.parquet",
]


# pos = {agent_id: (lon, lat) for agent_id, lon, lat in zip(agents_data['id'], agents_data['lon'], agents_data['lat'])}
agents_data = pandas.read_parquet(agents_datapath)
agents_data = agents_data[agents_data["area"] == 146100]

interactions_data = [
    pandas.read_parquet(file_path) for file_path in interactions_datapath
]

# Combine the DataFrames vertically
interactions_data = pandas.concat(
    interactions_data, ignore_index=True
).drop_duplicates()
interactions_data = interactions_data[
    (interactions_data["id_x"].isin(agents_data["id"]))
    & (interactions_data["id_y"].isin(agents_data["id"]))
]

# interactions_data = interactions_data.sample(300)

all_ids = list(interactions_data["id_x"].unique()) + list(
    interactions_data["id_x"].unique()
)

map_ids = {}
for i, proc_id in enumerate(all_ids):
    map_ids[proc_id] = i

interactions_data["id_x"] = interactions_data["id_x"].map(map_ids)
interactions_data["id_y"] = interactions_data["id_y"].map(map_ids)

edge_index = (
    torch.tensor(interactions_data[["id_x", "id_y"]].values, dtype=torch.long)
    .t()
    .contiguous()
)
# edge_index = edge_index / edge_index.max()
edge_attr = torch.tensor(interactions_data[["spec"]].values, dtype=torch.int)
data = Data(edge_index=edge_index, edge_attr=edge_attr)

G = to_networkx(data, to_undirected=True)
# pos = nx.spring_layout(G)
_, ax = plt.subplots()
# nx.draw(G, ax=ax, node_size=3, alpha=0.5)
nx.draw_spring(G, ax=ax, node_size=5, alpha=0.3)

plt.savefig("test.png")
plt.close()
