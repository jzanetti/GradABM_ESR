from logging import getLogger
from random import uniform as random_uniform

from torch import gather as torch_gather
from torch import manual_seed as torch_seed
from torch import nonzero as torch_nonzero
from torch import ones_like as torch_ones_like
from torch import randperm as torch_randperm
from torch import zeros_like as torch_zeros_like
from torch_geometric.nn import MessagePassing

from model import TORCH_SEED_NUM
from utils.utils import create_random_seed

logger = getLogger()


def infected_case_isolation(
    infected_idx, isolation_compliance_rate, isolation_intensity, t, min_cases: int = 10
):
    """Create infected case isolation scaling factor

    Args:
        infected_idx: Infected case index

    Returns:
        _type_: Scaling factor for infected case
    """
    if (isolation_compliance_rate) is None or (t == 0):
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

    isolated_agents_length = int(isolation_compliance_rate * len(infected_agents_index))

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
    R,
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
    S_A_s = (
        SFSusceptibility_age[x_i[:, 0].long()]
        * SFSusceptibility_sex[x_i[:, 1].long()]
        * SFSusceptibility_ethnicity[x_i[:, 2].long()]
        * SFSusceptibility_vaccine[x_i[:, 3].long()]
    )  # age * sex * ethnicity dependant * vaccine

    if t in [2, 3]:
        x = 3

    A_s_i = SFInfector[x_j[:, 4].long()]  # stage dependant

    B_n = edge_attr[1, :]
    integrals = torch_zeros_like(B_n)
    # infected_idx = x_j[:, 5].bool()
    infected_idx = x_j[:, 4].long() == 2.0
    infected_times = t - x_j[infected_idx, 6]

    integrals[infected_idx] = lam_gamma_integrals[infected_times.long()]

    # Isolate infected cases
    isolated_sf = infected_case_isolation(
        infected_idx,
        outbreak_ctl_cfg["isolation"]["compliance_rate"],
        outbreak_ctl_cfg["isolation"]["isolation_sf"],
        t,
    )

    edge_network_numbers = edge_attr[
        0, :
    ]  # to account for the fact that mean interactions start at 4th position of x
    I_bar = torch_gather(x_i[:, 7:30], 1, edge_network_numbers.view(-1, 1).long()).view(-1)

    if perturbation_flag:
        R = random_uniform(R * 0.7, R * 1.3)

    res = R * S_A_s * A_s_i * B_n * integrals * isolated_sf / I_bar  # Edge attribute 1 is B_n

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
        r0_value_trainable,
        lam_gamma_integrals,
        outbreak_ctl_cfg,
        perturbation_flag,
        vis_debug=False,
    ):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        t = data.t

        if vis_debug:
            self.vis_debug_graph(edge_index)

        return self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            t=t,
            R=r0_value_trainable,
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
        R,
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
            R,
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
