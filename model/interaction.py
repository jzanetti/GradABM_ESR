from pandas import read_csv as pandas_read_csv
from torch import device as torch_device
from torch import hstack as torch_hstack
from torch import ones as torch_ones
from torch import split as torch_split
from torch import tensor as torch_tensor
from torch import vstack as torch_vstack
from yaml import safe_load as yaml_load

from model import B_n


def create_interactions(
    interaction_cfg_path: str,
    interaction_graph_path: str,
    num_agents: int,
    device=torch_device("cpu"),
    network_type_dict: dict = {"company": 0, "school": 1},
) -> dict:
    """Obtain interaction networks

    Args:
        interaction_cfg_path (str): Network interaction configuration
        interaction_graph_path (str): Network graph nodes and edges
        num_agents (int): Number of agents.
        device (_type_, optional): Device type, e.g., CPU or CUDA. Defaults to torch_device("cpu").
        network_type_dict (dict, optional): Network to be used. Defaults to {"company": 0, "school": 1}.

    Returns:
        dict: The dict contains network information
    """
    agents_mean_interactions = 0 * torch_ones(num_agents, len(network_type_dict)).to(device)
    network_type_dict_inv = {value: key for key, value in network_type_dict.items()}

    with open(interaction_cfg_path, "r") as stream:
        mean_interaction_cfg = yaml_load(stream)

    for network_index in network_type_dict_inv:
        proc_interaction_mu = (
            torch_tensor(mean_interaction_cfg[network_type_dict_inv[network_index]]["mu"])
            .float()
            .to(device)
        )

        agents_mean_interactions[:, network_index] = proc_interaction_mu

    agents_mean_interactions_split = list(torch_split(agents_mean_interactions, 1, dim=1))
    agents_mean_interactions_split = [a.view(-1) for a in agents_mean_interactions_split]

    all_edgelist, all_edgeattr = init_interaction_graph(
        interaction_graph_path, network_type_dict, device
    )

    return {
        "agents_mean_interactions": agents_mean_interactions,
        "agents_mean_interactions_split": agents_mean_interactions_split,
        "network_type_dict": network_type_dict,
        "network_type_dict_inv": network_type_dict_inv,
        "all_edgelist": all_edgelist,
        "all_edgeattr": all_edgeattr,
    }


def init_interaction_graph(interaction_graph_path: str, network_type_dict: dict, device):
    """Initialize interaction graph

    Args:
        interaction_graph_path (str): Interaction graph path
        network_type_dict (dict): Network setup
        device (_type_): Device type, e.g., cpu

    Returns:
        _type_: _description_
    """
    random_network_edgelist_forward = (
        torch_tensor(pandas_read_csv(interaction_graph_path, header=None).to_numpy()).t().long()
    )
    random_network_edgelist_backward = torch_vstack(
        (random_network_edgelist_forward[1, :], random_network_edgelist_forward[0, :])
    )
    random_network_edgelist = torch_hstack(
        (random_network_edgelist_forward, random_network_edgelist_backward)
    )
    random_network_edgeattr_type = (
        torch_ones(random_network_edgelist.shape[1]).long() * network_type_dict["school"]
    )

    random_network_edgeattr_B_n = (
        torch_ones(random_network_edgelist.shape[1]).float() * B_n["school"]
    )
    random_network_edgeattr = torch_vstack(
        (random_network_edgeattr_type, random_network_edgeattr_B_n)
    )

    all_edgelist = torch_hstack((random_network_edgelist,))
    all_edgeattr = torch_hstack((random_network_edgeattr,))

    all_edgelist = all_edgelist.to(device)
    all_edgeattr = all_edgeattr.to(device)

    return all_edgelist, all_edgeattr
