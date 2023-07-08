from pandas import read_csv as pandas_read_csv
from torch import device as torch_device
from torch import hstack as torch_hstack
from torch import ones as torch_ones
from torch import split as torch_split
from torch import tensor as torch_tensor
from torch import vstack as torch_vstack
from yaml import safe_load as yaml_load


def agent_interaction_wrapper(
    agents_mean_interaction_cfg: dict,
    network_type_dict_inv: dict,
    num_agents: int,
    network_types: list,
    device,
) -> dict:
    """Obtain agents interaction intensity

    Args:
        agents_mean_interaction_cfg (dict): Agent interaction configuration
        num_agents (int): Number of agents
        network_types (list): Network types
        device (_type_): Device type, e.g., cpu

    Returns:
        dict: The dict contains agents interactions
    """
    agents_mean_interactions_mu = 0 * torch_ones(num_agents, len(network_types)).to(device)
    for network_index in network_type_dict_inv:
        proc_interaction_mu = (
            torch_tensor(agents_mean_interaction_cfg[network_type_dict_inv[network_index]]["mu"])
            .float()
            .to(device)
        )

        agents_mean_interactions_mu[:, network_index] = proc_interaction_mu

    agents_mean_interactions_mu_split = list(torch_split(agents_mean_interactions_mu, 1, dim=1))
    agents_mean_interactions_mu_split = [a.view(-1) for a in agents_mean_interactions_mu_split]

    return {
        "agents_mean_interactions_mu": agents_mean_interactions_mu,
        "agents_mean_interactions_mu_split": agents_mean_interactions_mu_split,
    }


def create_interactions(
    interaction_cfg_path: str,
    interaction_graph_path: str,
    num_agents: int,
    device=torch_device("cpu"),
    network_types: list = ["school", "household"],
) -> dict:
    """Obtain interaction networks

    Args:
        interaction_cfg_path (str): Network interaction configuration
        interaction_graph_path (str): Network graph nodes and edges
        num_agents (int): Number of agents.
        device (_type_, optional): Device type, e.g., CPU or CUDA. Defaults to torch_device("cpu").
        network_types (list, optional): Network to be used. Defaults to ["household", "school"].

    Returns:
        dict: The dict contains network information
    """
    # ----------------------------
    # Get basic network information
    # ----------------------------
    network_type_dict = {}
    for i, proc_type in enumerate(network_types):
        network_type_dict[proc_type] = i

    network_type_dict_inv = {value: key for key, value in network_type_dict.items()}

    # ----------------------------
    # Get agents interaction intensity
    # ----------------------------
    with open(interaction_cfg_path, "r") as stream:
        agents_mean_interaction_cfg = yaml_load(stream)

    agent_interaction_data = agent_interaction_wrapper(
        agents_mean_interaction_cfg, network_type_dict_inv, num_agents, network_types, device
    )

    # ----------------------------
    # Get edges interaction intensity
    # ----------------------------
    edges_mean_interaction_cfg = pandas_read_csv(interaction_graph_path, header=None)

    agents_mean_interactions_bn = {}
    for network_name in network_type_dict:
        agents_mean_interactions_bn[network_type_dict[network_name]] = agents_mean_interaction_cfg[
            network_name
        ]["bn"]

    all_edgelist, all_edgeattr = init_interaction_graph(
        edges_mean_interaction_cfg, network_type_dict, agents_mean_interactions_bn, device
    )

    return {
        "agents_mean_interactions_mu": agent_interaction_data["agents_mean_interactions_mu"],
        "agents_mean_interactions_mu_split": agent_interaction_data[
            "agents_mean_interactions_mu_split"
        ],
        "network_type_dict": network_type_dict,
        "network_type_dict_inv": network_type_dict_inv,
        "all_edgelist": all_edgelist,
        "all_edgeattr": all_edgeattr,
    }


def init_interaction_graph(
    interaction_graph_cfg: dict, network_type_dict: dict, agents_mean_interactions_bn: dict, device
):
    """Initialize interaction graph

    Args:
        interaction_graph_path (str): Interaction graph path
        network_type_dict (dict): Network setup
        agents_mean_interactions_bn (dict): The scale-factor for the network on which the interaction occured
        device (_type_): Device type, e.g., cpu

    Returns:
        _type_: _description_
    """
    random_network_edgelist_forward = (
        torch_tensor(interaction_graph_cfg.to_numpy()[:, 0:3]).t().long()
    )
    random_network_edgelist_backward = torch_vstack(
        (
            random_network_edgelist_forward[1, :],
            random_network_edgelist_forward[0, :],
            random_network_edgelist_forward[2, :],
        )
    )
    random_network_edge_all = torch_hstack(
        (random_network_edgelist_forward, random_network_edgelist_backward)
    )
    random_network_edgelist = random_network_edge_all[0:2, :]

    random_network_edgeattr_type = random_network_edge_all[2, :]

    random_network_edgeattr_B_n = [
        agents_mean_interactions_bn[key] for key in random_network_edgeattr_type.tolist()
    ]

    random_network_edgeattr_B_n = torch_tensor(random_network_edgeattr_B_n)
    # random_network_edgeattr_B_n = (
    #    torch_ones(random_network_edgelist.shape[1]).float()
    #    * agents_mean_interactions_bn["school"]
    # )
    random_network_edgeattr = torch_vstack(
        (random_network_edgeattr_type, random_network_edgeattr_B_n)
    )

    all_edgelist = torch_hstack((random_network_edgelist,))
    all_edgeattr = torch_hstack((random_network_edgeattr,))

    all_edgelist = all_edgelist.to(device)
    all_edgeattr = all_edgeattr.to(device)

    return all_edgelist, all_edgeattr
