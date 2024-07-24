import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

"""
GCNConv stands for Graph Convolutional Network Convolution. It’s a type of layer used in Graph Neural Networks (GNNs).

In a very simplified way, you can think of GCNConv as a tool that helps the nodes (or points) in a graph talk to their neighbors.

Imagine you’re at a party and you want to know more about the other guests. 
You could go around and ask each person directly, but that would take a lot of time. 
Instead, you decide to ask your immediate friends about the people they’ve met. 
This way, you get a summary of information about many guests quickly.

This is similar to what GCNConv does: 
    For each node in the graph, it aggregates information from its immediate neighbors. 
    This aggregated information is then used to update the node’s own information or features.

In the context of our Python code, GCNConv takes the feature matrix X 
    and the adjacency matrix edge_index (which tells us who are neighbors in the graph), 
    and it updates the features of each node based on its neighbors’ features.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# Define a simple GNN
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(1, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.sigmoid(self.conv1(x, edge_index))
        return x


# Create a simple graph with 3 nodes and 1 feature per node
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor(
    [[1], [0], [0]], dtype=torch.float
)  # Only the first node is infected initially

# Create a PyTorch Geometric data object
data = Data(x=x, edge_index=edge_index)

# Create a simple GNN
model = Net()

# Define the truth data as a time series
truth = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]])

# Define a simple loss function and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Train the GNN
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)

    # Update the health status of the node specified by the truth data
    data.x = truth[epoch % len(truth)].unsqueeze(1).float()

    # Compute the loss based on the current state of the truth data
    target = truth[epoch % len(truth)].unsqueeze(1).float()
    loss = loss_fn(out, target)

    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch}, Loss: {loss.item()}")

    # Print the health status of each node
    print(f"Health status after epoch {epoch}: {data.x.tolist()}")
