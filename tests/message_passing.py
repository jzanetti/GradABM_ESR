import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

"""
In this example, we create a custom MyMessagePassing class that extends the MessagePassing module. Inside the forward method, we perform the necessary preprocessing steps on the graph data, such as adding self-loops to the edge index and calculating the node degrees.

The message method defines the message function, which determines how information is passed from the source nodes to the target nodes during message passing. In this case, we simply return the node features of the source nodes (x_j).

The update method defines the update function, which aggregates the received messages and updates the node features accordingly. Here, we pass the aggregated output (aggr_out) as is.

Finally, we instantiate an instance of MyMessagePassing and pass our input node features (x) and edge index (edge_index) to the module's forward method. The resulting output is the updated node features after the message passing operation.

Note that this is a simplified example, and in real-world scenarios, you would typically include additional learnable parameters and more complex operations within the message and update methods to capture richer interactions between the nodes in the graph.
"""


class MyMessagePassing(MessagePassing):
    def __init__(self):
        super(MyMessagePassing, self).__init__(aggr="add")

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Calculate node degrees
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        # Normalize node features
        x = x * deg_inv_sqrt.unsqueeze(1)

        # Perform message passing
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j, edge_index, size):
        """x_i and x_j represent the node features of two connected nodes in a graph during the message passing process.
        Let's break down the code and explain it using a simple example.

        First, let's consider the input graph defined by the x tensor and edge_index tensor:

        x = torch.tensor([[1], [2], [3]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

        Here, we have three nodes with features [1], [2], and [3], respectively.
        The edge_index tensor indicates the connections between nodes.
        For example, the first edge connects node 0 to node 1, the second edge connects node 1 to node 0, and so on.

        Now, let's examine the message method in the MyMessagePassing class:

        During the message passing process, this method is called for each edge in the graph. It takes several arguments:

        - x_i: This represents the feature of the node sending the message. In other words, it corresponds to the source node feature for a particular edge.
        - x_j: This represents the feature of the node receiving the message. It corresponds to the target node feature for a particular edge.
        - edge_index: This tensor contains the indices of the source and target nodes for each edge.
        - size: This is the size of the graph, specified as (x.size(0), x.size(0)), where x.size(0) represents the number of nodes in the graph.

        """
        # Define the message function
        x = 3
        return x_j

    def update(self, aggr_out):
        # Define the update function
        return aggr_out


# Create a toy graph
x = torch.tensor([[1], [2], [3]], dtype=torch.float)
edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

# Instantiate the message passing module
model = MyMessagePassing()

# Perform message passing on the graph
output = model(x, edge_index)
print(output)
