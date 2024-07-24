# https://medium.com/mlearning-ai/ultimate-guide-to-graph-neural-networks-1-cora-dataset-37338c04fe6f
# https://pytorch-geometric-temporal.readthedocs.io/en/latest/notes/introduction.html#epidemiological-forecasting

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

# 136000: Eden Terrace, 133200: Queen Street
agents_datapath = "/tmp/gradabm_esr/Auckland_2019_measles3/input/agents.parquet"
interactions_datapath = "/tmp/gradabm_esr/Auckland_2019_measles3/input/interaction_graph_cfg_member_0_0.parquet"

# pos = {agent_id: (lon, lat) for agent_id, lon, lat in zip(agents_data['id'], agents_data['lon'], agents_data['lat'])}
agents_data = pandas.read_parquet(agents_datapath)
interactions_data = pandas.read_parquet(interactions_datapath)
# address_data = pandas.read_parquet(address_datapath)


# agents_data = agents_data[agents_data["area_work"] == 133200]
# interactions_data = interactions_data[interactions_data["id_x"].isin(agents_data["id"])]
interactions_data = interactions_data[interactions_data["spec"] == 4]


interactions_data = interactions_data.sample(3000, random_state=100)
# interactions_data["spec"] = interactions_data["spec"].astype(int)
edge_index = (
    torch.tensor(interactions_data[["id_x", "id_y"]].values, dtype=torch.long)
    .t()
    .contiguous()
)
edge_index = edge_index / edge_index.max()
edge_attr = torch.tensor(interactions_data[["spec"]].values, dtype=torch.int)
data = Data(edge_index=edge_index, edge_attr=edge_attr)

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
