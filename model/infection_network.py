from logging import getLogger
from random import uniform as random_uniform

from numpy import intersect1d as numpy_intersect1d
from numpy import isin as numpy_isin
from numpy import ones as numpy_ones
from numpy import where as numpy_where
from torch import gather as torch_gather
from torch import manual_seed as torch_seed
from torch import nonzero as torch_nonzero
from torch import ones_like as torch_ones_like
from torch import randperm as torch_randperm
from torch import tensor as torch_tensor
from torch import zeros_like as torch_zeros_like
from torch_geometric.nn import MessagePassing

from input import LOC_INDEX
from model import DEVICE, TORCH_SEED_NUM
from utils.utils import create_random_seed

logger = getLogger()


def school_closure(infected_idx, edge_attr, school_closure_cfg):
    school_closure_sf = numpy_ones(len(infected_idx))

    if not school_closure_cfg["enable"]:
        return torch_tensor(school_closure_sf).to(DEVICE)

    infected_idx_index = numpy_where(infected_idx.cpu().detach().numpy() == True)[0]

    if len(infected_idx_index) > 0:
        school_index = numpy_where(edge_attr[0, :].cpu().detach().numpy() == LOC_INDEX["school"])[
            0
        ]

        overlap_index = numpy_intersect1d(infected_idx_index, school_index)
        schools_to_shutdown = edge_attr[2, :][overlap_index].unique().cpu().detach()

        potential_indices = numpy_where(
            numpy_isin(edge_attr[2].cpu().detach().numpy(), schools_to_shutdown)
        )[0]
        schools_to_shutdown_indices = numpy_intersect1d(school_index, potential_indices)
        school_closure_sf[schools_to_shutdown_indices] *= school_closure_cfg["scaling_factor"]

    return torch_tensor(school_closure_sf).to(DEVICE)


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
    if (isolation_compliance_rate is None) or (contact_tracing_coverage is None) or (t == 0):
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

    identified_infected_agents_length = infected_agents_length * contact_tracing_coverage
    isolated_agents_length = int(isolation_compliance_rate * identified_infected_agents_length)

    if TORCH_SEED_NUM is not None:
        torch_seed(TORCH_SEED_NUM["isolation_policy"])

    isolated_agents_index = torch_randperm(infected_agents_length)[:isolated_agents_length]
    isolated_mask = torch_zeros_like(infected_agents)
    isolated_mask[infected_agents_index[isolated_agents_index]] = 1.0 - isolation_intensity

    isolated_sf = 1.0 - isolated_mask

    return isolated_sf


def lam(
    x_i,
    x_j,
    edge_attr,
    t,
    vaccine_efficiency_spread,
    contact_tracing_coverage,
    SFSusceptibility_age,
    SFSusceptibility_sex,
    SFSusceptibility_ethnicity,
    SFSusceptibility_vaccine,
    SFInfector,
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
    *self.agents_mean_interactions_mu_split,  # 7 to 29: represents the number of venues where agents can interact with each other
    torch.arange(self.params["num_agents"]).to(self.device),  # Agent ids (30)
    """

    # --------------------------------------------------
    # Step 1:
    # This section calculates S_A_s, which appears
    # to represent a susceptibility factor for each edge.
    # It seems to depend on the age, sex, ethnicity,
    # and vaccination status of the source node x_i.
    # --------------------------------------------------
    if t == -1:
        SFSusceptibility_vaccine = torch_ones_like(x_i[:, 3].long())
    else:
        SFSusceptibility_vaccine = vaccine_efficiency_spread * x_i[:, 3].long()
    S_A_s = (
        SFSusceptibility_age[x_i[:, 0].long()]
        * SFSusceptibility_sex[x_i[:, 1].long()]
        * SFSusceptibility_ethnicity[x_i[:, 2].long()]
        * SFSusceptibility_vaccine
    )  # age * sex * ethnicity dependant * vaccine

    # --------------------------------------------------
    # Step 2:
    # A_s_i is calculated based on the stage (x_j[:, 4]) of the target node x_j.
    #  It seems to represent an infectivity factor related to the stage.
    # --------------------------------------------------
    A_s_i = SFInfector[x_j[:, 4].long()]  # stage dependant

    B_n = edge_attr[1, :]
    integrals = torch_zeros_like(B_n)
    # infected_idx = x_j[:, 5].bool()
    # infected_idx = x_j[:, 4].long() == 4.0
    # infected_idx_length = infected_idx.tolist().count(True)
    # if infected_idx_length == 0:
    infected_idx = x_j[:, 4].long() == 2.0

    infected_times = t - x_j[infected_idx, 6]
    # infected_idx2 = x_j[:, 4].long() == 4.0
    # print(infected_idx.tolist().count(True), infected_idx2.tolist().count(True))

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

    edge_network_numbers = edge_attr[
        0, :
    ]  # to account for the fact that mean interactions start at 4th position of x
    I_bar = torch_gather(x_i[:, 7:30], 1, edge_network_numbers.view(-1, 1).long()).view(-1)

    if perturbation_flag:
        R = random_uniform(0.7, 1.3)
    else:
        R = 1.0

    res = (
        R * S_A_s * A_s_i * B_n * integrals * isolated_sf * school_closure_sf / I_bar
    )  # Edge attribute 1 is B_n

    # print(t, integrals[infected_idx].sum(), res.sum())

    return res.view(-1, 1)


class InfectionNetwork(MessagePassing):
    """Contact network with graph message passing function for disease spread"""

    def __init__(
        self,
        SFSusceptibility_age,
        SFSusceptibility_sex,
        SFSusceptibility_ethnicity,
        SFSusceptibility_vaccine,
        SFInfector,
        device,
    ):
        super(InfectionNetwork, self).__init__(aggr="add")
        self.lam = lam
        self.SFSusceptibility_age = SFSusceptibility_age
        self.SFSusceptibility_sex = SFSusceptibility_sex
        self.SFSusceptibility_ethnicity = SFSusceptibility_ethnicity
        self.SFSusceptibility_vaccine = SFSusceptibility_vaccine
        self.SFInfector = SFInfector
        # self.lam_gamma_integrals = lam_gamma_integrals
        self.device = device

    def forward(
        self,
        data,
        lam_gamma_integrals,
        outbreak_ctl_cfg,
        perturbation_flag,
        vis_debug=False,
    ):
        x = data.x
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
            SFSusceptibility_age=self.SFSusceptibility_age,
            SFSusceptibility_sex=self.SFSusceptibility_sex,
            SFSusceptibility_ethnicity=self.SFSusceptibility_ethnicity,
            SFSusceptibility_vaccine=self.SFSusceptibility_vaccine,
            SFInfector=self.SFInfector,
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
        SFSusceptibility_age,
        SFSusceptibility_sex,
        SFSusceptibility_ethnicity,
        SFSusceptibility_vaccine,
        SFInfector,
        lam_gamma_integrals,
        outbreak_ctl_cfg,
        perturbation_flag,
    ):
        # x_j has shape [E, in_channels]
        tmp = self.lam(
            x_i,
            x_j,
            edge_attr,
            t,
            vaccine_efficiency_spread,
            contact_tracing_coverage,
            SFSusceptibility_age,
            SFSusceptibility_sex,
            SFSusceptibility_ethnicity,
            SFSusceptibility_vaccine,
            SFInfector,
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
        pos = spring_layout(subgraph)  # Positions the nodes using the spring layout algorithm

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
