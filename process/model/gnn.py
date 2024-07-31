from logging import getLogger
from random import uniform as random_uniform

from torch import cat as torch_cat
from torch import int8 as torch_int8
from torch import ones_like as torch_ones_like
from torch import tensor as torch_tensor
from torch import vstack as torch_vstack
from torch import zeros_like as torch_zeros_like
from torch_geometric.nn import MessagePassing

from process import ETHNICITY_INDEX
from process.model import STAGE_INDEX
from process.model.policy import infected_case_isolation, school_closure

logger = getLogger()


def lam(
    x_i,
    x_j,
    edge_attr,
    t,
    total_timesteps,
    vaccine_efficiency_spread,
    contact_tracing_coverage,
    scaling_factor,
    lam_gamma_integrals,
    outbreak_ctl_cfg,
    perturbation_flag,
):
    """
    self.agents_id,  # 0: id
    self.agents_ages,  # 1: age
    self.agents_sex,  # 2: sex
    self.agents_ethnicity,  # 3: ethnicity
    self.agents_vaccine,  # 4: vaccine
    self.current_stages.detach(),  # 5: stage
    self.agents_infected_index.to(self.device),  # 6: infected index
    self.agents_infected_time.to(self.device),  # 7: infected time

    x_i: node receive information;
    x_j: node send information
    """
    # --------------------------------------------------
    # Step 1 (Target Susceptible by vaccine):
    # --------------------------------------------------
    vaccine_index = x_i[:, 4].long()
    if t == -1:
        scaling_factor_vaccine = torch_ones_like(vaccine_index)
    else:
        scaling_factor_vaccine_yes = (
            vaccine_efficiency_spread * (x_i[:, 4] == 1.0).float()
        )
        scaling_factor_vaccine_no = 1.0 * (x_i[:, 4] == 0.0).float()
        scaling_factor_vaccine = scaling_factor_vaccine_yes + scaling_factor_vaccine_no

    # -------------------------------------------------
    # Step 2 (Target Susceptible by age, gender, ethnicity):
    # -------------------------------------------------
    scaling_factor_age_gender_ethnicity = (
        scaling_factor["age"][x_i[:, 1].long()]
        * scaling_factor["gender"][x_i[:, 2].long()]
        * scaling_factor["ethnicity"][x_i[:, 3].long()]
    )

    # --------------------------------------------------
    # Step 3 (Source Infectiousness by age, gender, ethnicity):
    # --------------------------------------------------
    scaling_factor_symptom = torch_ones_like(scaling_factor_age_gender_ethnicity)
    scaling_factor_symptom_infected = (
        3.0 * (x_j[:, 5] == STAGE_INDEX["infected"]).float()
    )
    scaling_factor_symptom_exposed = (
        0.75 * (x_j[:, 5] == STAGE_INDEX["exposed"]).float()
    )
    scaling_factor_symptom_others = (
        1.0
        * (
            (x_j[:, 5] != STAGE_INDEX["infected"])
            & (x_j[:, 5] != STAGE_INDEX["exposed"])
        ).float()
    )
    scaling_factor_symptom = (
        scaling_factor_symptom_others
        + scaling_factor_symptom_infected
        + scaling_factor_symptom_exposed
    )

    # --------------------------------------------------
    # Step 4 (Source Infectiousness by infection time):
    # --------------------------------------------------
    scaling_factor_infection_time = torch_zeros_like(scaling_factor_symptom)
    infected_idx = x_j[:, 6].long().bool()
    infected_times = t - x_j[infected_idx, 7]
    scaling_factor_infection_time[infected_idx] = lam_gamma_integrals[
        infected_times.long()
    ]

    # --------------------------------------------------
    # Step 5: Outbreak measures
    # --------------------------------------------------
    # Isolate infected cases
    if t == -1:
        isolated_sf = torch_ones_like(infected_idx)
        school_closure_sf = torch_ones_like(infected_idx)
    else:
        isolated_sf = infected_case_isolation(
            infected_idx,
            contact_tracing_coverage,
            outbreak_ctl_cfg["isolation"],
            t,
        )

        # Shutdown school
        school_closure_sf = school_closure(
            infected_idx, edge_attr, outbreak_ctl_cfg["school_closure"]
        )

    res = (
        scaling_factor_vaccine
        # vaccine_efficiency_spread
        * scaling_factor_age_gender_ethnicity
        * scaling_factor_symptom
        * scaling_factor_infection_time
        * edge_attr[2, :]
        # * integrals
        * isolated_sf
        * school_closure_sf
        / edge_attr[1, :]
    )  # Edge attribute 1 is B_n

    return res.view(-1, 1)
    # return torch_cat((x_j[:, 0].view(-1, 1), res.view(-1, 1)), dim=1)


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
        total_timesteps = data.total_timesteps
        vaccine_efficiency_spread = data.vaccine_efficiency_spread
        contact_tracing_coverage = data.contact_tracing_coverage

        if vis_debug:
            self.vis_debug_graph(edge_index)

        return self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            t=t,
            total_timesteps=total_timesteps,
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
        total_timesteps,
        vaccine_efficiency_spread,
        contact_tracing_coverage,
        scaling_factor,
        lam_gamma_integrals,
        outbreak_ctl_cfg,
        perturbation_flag,
    ):
        """By default, We generally refer to x_i as the nodes which aggregate information,
        and to x_j as the nodes which send information along the edges.

        The default message passing flow is source_to_target:
        so x_i refers to the targets and x_j refers to the source nodes.
        If you want to change this behavior, you can change the message passing flow via flow="target_to_source".

        See details: https://github.com/pyg-team/pytorch_geometric/issues/699
        """
        tmp = self.lam(
            x_i,
            x_j,
            edge_attr,
            t,
            total_timesteps,
            vaccine_efficiency_spread,
            contact_tracing_coverage,
            scaling_factor,
            lam_gamma_integrals,
            outbreak_ctl_cfg,
            perturbation_flag,
        )
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
