# https://medium.com/mlearning-ai/ultimate-guide-to-graph-neural-networks-1-cora-dataset-37338c04fe6f
# https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html#epidemiological-forecasting

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
# address_data = pandas.read_parquet(address_datapath)
# agents_data = agents_data[agents_data["school"] == "139100_secondary_0"]

# agents_data = agents_data[agents_data["area_work"] == 133200]
# interactions_data = interactions_data[interactions_data["id_x"].isin(agents_data["id"])]
interactions_data = interactions_data[interactions_data["spec"] == 5]
# interactions_data = interactions_data[
#    interactions_data["id_x"].isin(agents_data["id"])
#    & interactions_data["id_y"].isin(agents_data["id"])
# ]

interactions_data = interactions_data.sample(10, random_state=100)
merged_df = pandas.merge(
    interactions_data,
    address_data,
    left_on="group",
    right_on="name",
    suffixes=("_X", "_Y"),
)

node_coordinates = {}
for row in merged_df.itertuples():
    node_id = row.id_x
    latitude = row.latitude
    longitude = row.longitude
    node_coordinates[node_id] = {"lat": latitude, "lon": longitude}


for row in merged_df.itertuples():
    node_id = row.id_y
    if node_id in node_coordinates:
        continue
    latitude = row.latitude
    longitude = row.longitude
    node_coordinates[node_id] = {"lat": latitude, "lon": longitude}

idx = []
idy = []
for i, row in merged_df.iterrows():
    idx.append(row["id_x"])
    idy.append(row["id_y"])

edge_index = torch.tensor(numpy.array([idx, idy]))
# Extract unique nodes from edge_index
unique_nodes = torch.unique(edge_index.flatten())

# Create pos tensor from node coordinates
pos = torch.tensor(
    [
        [node_coordinates[int(node)]["lat"], node_coordinates[int(node)]["lon"]]
        for node in unique_nodes
    ],
    dtype=torch.float,
)


# Add a random value between -0.0001 and 0.0001 to latitude and longitude
# merged_df["latitude"] += numpy.random.uniform(0001, len(merged_df))
# merged_df["longitude"] += numpy.random.uniform(-0.0001, 0.0001, len(merged_df))

idx = []
idy = []
for i, row in merged_df.iterrows():
    idx.append(row["id_x"])
    idy.append(row["id_y"])

data = Data(edge_index=torch.tensor(numpy.array([idx, idy])), pos=pos)
G = to_networkx(data, to_undirected=False)

pos = {}
for proc_node in enumerate(G.nodes):
    pos[proc_node] = idy_latlon

pos3 = nx.spring_layout(G, seed=42)

nx.draw_networkx(G, pos, width=0.3, alpha=0.5)


# edge_index = (
#    torch.tensor(merged_df[["id_x", "id_y"]].values, dtype=torch.long).t().contiguous()
# )
# edge_index = edge_index / edge_index.max()
# data = Data(edge_index=edge_index)
# G = to_networkx(data, to_undirected=False)
# pos = nx.spring_layout(G, seed=42)
# Assuming you have the merged DataFrame 'merged_df'
# G = nx.Graph()

# Add nodes with latitude and longitude attributes
"""
for _, row in merged_df.iterrows():
    if row["id_x"] == 349012:
        x = 3
    G.add_node(row["id_x"], latitude=row["latitude"], longitude=row["longitude"])

# Add edges (assuming you have edge information in 'merged_df')
for _, row in merged_df.iterrows():
    G.add_edge(row["id_x"], row["id_y"])

pos = {}
for node in G.nodes():
    try:
        pos[node] = (G.nodes[node]["longitude"], G.nodes[node]["latitude"])
    except KeyError:
        pass
"""
"""
# interactions_data["spec"] = interactions_data["spec"].astype(int)
edge_index = (
    torch.tensor(interactions_data[["id_x", "id_y"]].values, dtype=torch.long)
    .t()
    .contiguous()
)
edge_index = edge_index / edge_index.max()
edge_attr = torch.tensor(interactions_data[["spec"]].values, dtype=torch.int)
data = Data(edge_index=edge_index, edge_attr=edge_attr)
print(data)
edge_colors = [
    EGDE_COLOR_INDEX[attr_value[0]][1] for attr_value in data.edge_attr.numpy()
]

G = to_networkx(data, to_undirected=False)
"""
# pos = nx.spring_layout(G, seed=42)
# pos = nx.nx_agraph.graphviz_layout(G)
nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.5)

# nx.draw_spring(G, width=0.3, alpha=0.5, node_size=5)

# for attr_value, color in EGDE_COLOR_INDEX.items():
#    plt.plot([], [], color=color[1], label=f"Attr {color[0]}")

plt.savefig("test.png")
plt.close()

"""
print("xxxx")
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split

loader = ChickenpoxDatasetLoader()

dataset = loader.get_dataset()

train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)


import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import DCRNN


class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features):
        super(RecurrentGCN, self).__init__()
        self.recurrent = DCRNN(node_features, 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h


from tqdm import tqdm

model = RecurrentGCN(node_features=4)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()

for epoch in tqdm(range(200)):
    cost = 0
    for time, snapshot in enumerate(train_dataset):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
    cost = cost / (time + 1)
    cost.backward()
    optimizer.step()
    optimizer.zero_grad()

model.eval()
cost = 0
for time, snapshot in enumerate(test_dataset):
    y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
    cost = cost + torch.mean((y_hat - snapshot.y) ** 2)
cost = cost / (time + 1)
cost = cost.item()
print("MSE: {:.4f}".format(cost))
"""
