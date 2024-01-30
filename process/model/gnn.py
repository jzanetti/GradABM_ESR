from logging import getLogger
from random import uniform as random_uniform

from torch import int8 as torch_int8
from torch import int16 as torch_int16
from torch import manual_seed as torch_seed
from torch import nonzero as torch_nonzero
from torch import ones_like as torch_ones_like
from torch import randperm as torch_randperm
from torch import tensor as torch_tensor
from torch import zeros_like as torch_zeros_like
from torch_geometric.nn import MessagePassing

from process.model import TORCH_SEED_NUM
from process.model.policy import school_closure

logger = getLogger()


def infected_case_isolation(
    infected_idx,
    contact_tracing_coverage,
    isolation_compliance_rate,
    isolation_intensity,
    t,
    min_cases: int = 10,
):
    """Create infected case isolation scaling factor

    Args:
        infected_idx: Infected case index

    Returns:
        _type_: Scaling factor for infected case
    """
    if (
        (isolation_compliance_rate is None)
        or (contact_tracing_coverage is None)
        or (t == 0)
    ):
        return torch_ones_like(infected_idx)

    # infected agents (0.0: uninfected; 1.0: infected)
    infected_agents = infected_idx.float()
    infected_agents_index = torch_nonzero(infected_agents == 1.0).squeeze()
    try:
        infected_agents_length = len(infected_agents_index)
    except TypeError:  # len() of a 0-d tensor
        infected_agents_length = -999.0

    if infected_agents_length < min_cases:
        return torch_ones_like(infected_agents)

    identified_infected_agents_length = (
        infected_agents_length * contact_tracing_coverage
    )
    isolated_agents_length = int(
        isolation_compliance_rate * identified_infected_agents_length
    )

    if TORCH_SEED_NUM is not None:
        torch_seed(TORCH_SEED_NUM["isolation_policy"])

    isolated_agents_index = torch_randperm(infected_agents_length)[
        :isolated_agents_length
    ]
    isolated_mask = torch_zeros_like(infected_agents)
    isolated_mask[infected_agents_index[isolated_agents_index]] = (
        1.0 - isolation_intensity
    )

    isolated_sf = 1.0 - isolated_mask

    return isolated_sf


def lam(
    x_i,
    x_j,
    edge_attr,
    t,
    vaccine_efficiency_spread,
    contact_tracing_coverage,
    scaling_factor,
    lam_gamma_integrals,
    outbreak_ctl_cfg,
    perturbation_flag,
):
    """
    self.agents_ages,  # 0: age
    self.agents_sex,  # 1: sex
    self.agents_ethnicity,  # 2: ethnicity
    self.agents_vaccine,  # 3: vaccine
    self.current_stages.detach(),  # 4: stage
    self.agents_infected_index.to(self.device),  # 5: infected index
    self.agents_infected_time.to(self.device),  # 6: infected time
    """

    # --------------------------------------------------
    # Step 1:
    # This section calculates S_A_s, which appears
    # to represent a susceptibility factor for each edge.
    # It seems to depend on the age, sex, ethnicity,
    # and vaccination status of the source node x_i.
    # --------------------------------------------------
    vaccine_index = x_i[:, 3].long()
    if t == -1:
        scaling_factor_vaccine = torch_ones_like(vaccine_index)
    else:
        scaling_factor_vaccine = (
            torch_tensor(
                scaling_factor["vaccine"].tolist(), device=vaccine_index.device
            )[vaccine_index]
            * vaccine_efficiency_spread
        )
    S_A_s = (
        scaling_factor["age"][x_i[:, 0].long()]
        * scaling_factor["gender"][x_i[:, 1].long()]
        * scaling_factor["ethnicity"][x_i[:, 2].long()]
        * scaling_factor_vaccine
    )  # age * sex * ethnicity dependant * vaccine

    # --------------------------------------------------
    # Step 2: A_s_i is calculated based on the stage (x_j[:, 4]) of the target node x_j.
    #  It seems to represent an infectivity factor related to the stage.
    # --------------------------------------------------
    A_s_i = scaling_factor["symptom"][x_j[:, 4].long()]  # stage dependant

    # --------------------------------------------------
    # Step 3: Scaling factor depending on the infection time
    # --------------------------------------------------
    integrals = torch_zeros_like(S_A_s)
    infected_idx = x_j[:, 4].long() == 2.0
    infected_times = t - x_j[infected_idx, 6]
    integrals[infected_idx] = lam_gamma_integrals[infected_times.long()]

    # Isolate infected cases
    if t == -1:
        isolated_sf = torch_ones_like(infected_idx)
        school_closure_sf = torch_ones_like(infected_idx)
    else:
        isolated_sf = infected_case_isolation(
            infected_idx,
            contact_tracing_coverage,
            outbreak_ctl_cfg["isolation"]["compliance_rate"],
            outbreak_ctl_cfg["isolation"]["isolation_sf"],
            t,
        )

        # Shutdown school
        school_closure_sf = school_closure(
            infected_idx, edge_attr, outbreak_ctl_cfg["school_closure"]
        )

    if perturbation_flag:
        R = random_uniform(0.7, 1.3)
    else:
        R = 1.0
    # import torch

    # print(t, f"test1: {round(torch.cuda.memory_allocated(0) / (1024**3), 3) } Gb")
    res = (
        R
        * S_A_s
        * A_s_i
        * edge_attr[2, :]
        * integrals
        * isolated_sf
        * school_closure_sf
        / edge_attr[1, :]
    )  # Edge attribute 1 is B_n

    # print(t, integrals[infected_idx].sum(), res.sum())

    return res.view(-1, 1)


class GNN_model(MessagePassing):
    """Contact network with graph message passing function for disease spread"""

    def __init__(
        self,
        scaling_factor_age,
        scaling_factor_gender,
        scaling_factor_ethnicity,
        scaling_factor_vaccine,
        scaling_factor_symptom,
        # device,
    ):
        super(GNN_model, self).__init__(aggr="add")
        self.lam = lam
        self.scaling_factor_age = scaling_factor_age
        self.scaling_factor_gender = scaling_factor_gender
        self.scaling_factor_ethnicity = scaling_factor_ethnicity
        self.scaling_factor_vaccine = scaling_factor_vaccine
        self.scaling_factor_symptom = scaling_factor_symptom
        # self.lam_gamma_integrals = lam_gamma_integrals
        # self.device = device

    def forward(
        self,
        data,
        lam_gamma_integrals,
        outbreak_ctl_cfg,
        perturbation_flag,
        vis_debug=False,
    ):
        x = data.x.to(dtype=torch_int8)
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        t = data.t
        vaccine_efficiency_spread = data.vaccine_efficiency_spread
        contact_tracing_coverage = data.contact_tracing_coverage

        if vis_debug:
            self.vis_debug_graph(edge_index)

        """
        The propagate() method of the MessagePassing class performs the 
        message passing algorithm on a given graph. It takes the following arguments:

         - x: A tensor of node features.
         - edge_index: A tensor of edge indices.

        The propagate() method returns a tensor of updated node features.

        The x_i and x_j tensors are the node features for the source and target nodes of each edge, respectively. 
        They are obtained by splitting the x tensor using the edge_index tensor.

        For example:

            edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
            x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float)
            
            - The above tensor could represent a graph with two nodes and two edges. 
            - The first row of the tensor represents the first edge, which goes from node 0 to node 1. 
            - The second row of the tensor represents the second edge, which goes from node 1 to node 2.
            
            The propagate() method would split the x tensor as follows:

            x_i = torch.tensor([[1.0], [2.0]])
            x_j = torch.tensor([[2.0], [3.0]]) 

            - The x_i tensor would then contain the node features for the source nodes of each edge:
            - The x_j tensor would then contain the node features for the target nodes of each edge:

        We also can get the source node where have the most connections (e.g., having the most target nodes):

            import torch
            num_outgoing_edges = torch.bincount(edge_index[0])
            max_outgoing_edges_idx = torch.argmax(num_outgoing_edges)

        """

        return self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            t=t,
            vaccine_efficiency_spread=vaccine_efficiency_spread,
            contact_tracing_coverage=contact_tracing_coverage,
            scaling_factor={
                "age": self.scaling_factor_age,
                "gender": self.scaling_factor_gender,
                "ethnicity": self.scaling_factor_ethnicity,
                "symptom": self.scaling_factor_symptom,
                "vaccine": self.scaling_factor_vaccine,
            },
            lam_gamma_integrals=lam_gamma_integrals,
            outbreak_ctl_cfg=outbreak_ctl_cfg,
            perturbation_flag=perturbation_flag,
        )

    def message(
        self,
        x_i,
        x_j,
        edge_attr,
        t,
        vaccine_efficiency_spread,
        contact_tracing_coverage,
        scaling_factor,
        lam_gamma_integrals,
        outbreak_ctl_cfg,
        perturbation_flag,
    ):
        tmp = self.lam(
            x_i,
            x_j,
            edge_attr,
            t,
            vaccine_efficiency_spread,
            contact_tracing_coverage,
            scaling_factor,
            lam_gamma_integrals,
            outbreak_ctl_cfg,
            perturbation_flag,
        )  # tmp has shape [E, 2 * in_channels]
        return tmp

    def vis_debug_graph(self, edge_index, source_node=304667):
        import matplotlib.pyplot as plt
        from networkx import Graph, bfs_edges, draw_networkx, spring_layout

        logger.info("Creating VIS graph debug:")
        edge_index_value = edge_index.cpu().T.numpy()
        G = Graph()
        logger.info("   - Adding edges from edge_index")
        G.add_edges_from(edge_index_value)

        logger.info("   - Creating connection nodes")
        connected_nodes = list(bfs_edges(G, source_node))
        connected_nodes = connected_nodes[0:3000]

        logger.info("   - Creating tree nodes")
        nodes_connected_to_0 = [source_node] + [v for u, v in connected_nodes]

        logger.info("   - Creating subgraph")
        # Create a subgraph containing only the nodes connected to node "0"
        subgraph = G.subgraph(nodes_connected_to_0)

        plt.figure(figsize=(10, 8))
        logger.info("   - Creating spring_layout")
        pos = spring_layout(
            subgraph
        )  # Positions the nodes using the spring layout algorithm

        logger.info("   - Creating visualization")
        plt.figure(figsize=(10, 8))
        draw_networkx(
            subgraph,
            pos,
            with_labels=False,
            node_size=10,
            node_color="skyblue",
            font_size=12,
            font_weight="bold",
            font_color="black",
        )
        print("step 4 ...")
        plt.title(
            "Graph visualization for the agent 304667 over a week \n the first 3000 possible graph nodes/edges.",
            fontsize=16,
        )

        plt.savefig("debug_graph_vis.png", bbox_inches="tight")
        plt.close()
