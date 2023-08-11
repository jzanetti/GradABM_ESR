import torch
from numpy import array
from numpy import float64 as numpy_float64
from pandas import DataFrame
from pandas import read_csv as pandas_read_csv
from pandas import read_parquet as pandas_read_parquet
from pandas import to_numeric as pandas_to_numeric
from torch import hstack as torch_hstack
from torch import ones as torch_ones
from torch import split as torch_split
from torch import tensor as torch_tensor
from torch import vstack as torch_vstack
from torch.utils.data.dataloader import DataLoader

from input import LOC_INDEX
from model import DEVICE


class SeqData(torch.utils.data.Dataset):
    def __init__(self, y):
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.y[idx]


def train_data_wrapper(
    y_path,
    target_cfg,
    batch_size: int = 1,
    shuffle: bool = True,
) -> DataLoader:
    """Create training data for parameters

    Args:
        y_path (str): Target data
        batch_size (int, optional): Training batch size. Defaults to 1.
        shuffle (bool, optional): If shuffle the data. Defaults to True.

    Returns:
        DataLoader: Torch data loader
    """

    def _read_cases(y_path: str) -> dict:
        """Read target cases

        Args:
            y_path (str): Target cases path

        Returns:
            dict: Y data
        """
        y_input = pandas_read_csv(y_path)[
            target_cfg["start_timestep"] : target_cfg["end_timestep"]
        ]
        y_input = y_input.to_numpy()
        y = []
        tensor_y = torch.from_numpy(array([y_input]).astype(numpy_float64))
        y.append(tensor_y)
        y_train = torch.cat(y, axis=0)

        return y_train

    y_data = _read_cases(y_path)

    train_dataset = SeqData(y_data)
    return {
        "target": torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        ).dataset.y.to(DEVICE),
        "total_timesteps": y_data.shape[1],
    }


def create_agents(agent_filepath: str, max_agents: int or None = None) -> DataFrame:
    """Create agents

    Args:
        agent_filepath (str): Agent data path

    Returns:
        DataFrame: Agents information
    """
    if agent_filepath.endswith("parquet"):
        all_agents = pandas_read_parquet(agent_filepath)
    elif agent_filepath.endswith("csv"):
        all_agents = pandas_read_csv(agent_filepath)
    else:
        raise Exception("The file format is not supported")

    if max_agents is not None:
        all_agents = all_agents.head(max_agents)

    return all_agents


def agent_interaction_wrapper(
    agents_mean_interaction_cfg: dict, network_type_dict_inv: dict, num_agents: int
) -> dict:
    """Obtain agents interaction intensity

    Args:
        agents_mean_interaction_cfg (dict): Agent interaction configuration
        num_agents (int): Number of agents

    Returns:
        dict: The dict contains agents interactions
    """
    agents_mean_interactions_mu = 0 * torch_ones(num_agents, len(LOC_INDEX)).to(DEVICE)
    for network_index in network_type_dict_inv:
        proc_interaction_mu = (
            torch_tensor(agents_mean_interaction_cfg[network_type_dict_inv[network_index]]["mu"])
            .float()
            .to(DEVICE)
        )

        agents_mean_interactions_mu[:, network_index] = proc_interaction_mu

    agents_mean_interactions_mu_split = list(torch_split(agents_mean_interactions_mu, 1, dim=1))
    agents_mean_interactions_mu_split = [a.view(-1) for a in agents_mean_interactions_mu_split]

    return {
        "agents_mean_interactions_mu": agents_mean_interactions_mu,
        "agents_mean_interactions_mu_split": agents_mean_interactions_mu_split,
    }


def create_interactions(
    interaction_cfg: dict, interaction_graph_path: str, num_agents: int
) -> dict:
    """Obtain interaction networks

    Args:
        interaction_cfg_path (str): Network interaction configuration
        interaction_graph_path (str): Network graph nodes and edges
        num_agents (int): Number of agents.
        device (_type_, optional): Device type, e.g., CPU or CUDA. Defaults to torch_device("cpu").
        network_types (list, optional): Network to be used. Defaults to ["household", "school"].

        loc_index = {
            "household": 0,
            "city_transport": 1,
            "inter_city_transport": 2,
            "gym": 3,
            "grocery": 4,
            "pub": 5,
            "cinema": 6,
            "school": 7,
            "company": 8,
        }

    Returns:
        dict: The dict contains network information
    """
    # ----------------------------
    # Get basic network information
    # ----------------------------
    network_type_dict = {}
    for proc_type in LOC_INDEX:
        network_type_dict[proc_type] = LOC_INDEX[proc_type]

    network_type_dict_inv = {value: key for key, value in network_type_dict.items()}

    # ----------------------------
    # Get agents interaction intensity
    # ----------------------------
    # with open(interaction_cfg_path, "r") as stream:
    #    agents_mean_interaction_cfg = yaml_load(stream)

    agent_interaction_data = agent_interaction_wrapper(
        interaction_cfg, network_type_dict_inv, num_agents
    )

    # ----------------------------
    # Get edges interaction intensity
    # ----------------------------
    if interaction_graph_path.endswith("csv"):
        create_bidirection = True
        edges_mean_interaction_cfg = pandas_read_csv(interaction_graph_path, header=None)
    if interaction_graph_path.endswith("parquet"):
        create_bidirection = False
        edges_mean_interaction_cfg = pandas_read_parquet(interaction_graph_path)

    counts_df = edges_mean_interaction_cfg.groupby("id_x")["id_y"].nunique().reset_index()
    counts_df.columns = ["id_x", "count_of_id_y"]

    agents_mean_interactions_bn = {}
    for network_name in network_type_dict:
        agents_mean_interactions_bn[network_type_dict[network_name]] = interaction_cfg[
            network_name
        ]["bn"]

    all_edgelist, all_edgeattr = init_interaction_graph(
        edges_mean_interaction_cfg, agents_mean_interactions_bn, create_bidirection
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
    interaction_graph_cfg: dict,
    agents_mean_interactions_bn: dict,
    create_bidirection: bool,
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

    interaction_graph_cfg = (
        interaction_graph_cfg.apply(pandas_to_numeric, errors="coerce").values[1:, :].astype(int)
    )

    if create_bidirection:
        random_network_edgelist_forward = torch_tensor(interaction_graph_cfg[:, 0:3]).t().long()
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
    else:
        random_network_edge_all = torch_tensor(interaction_graph_cfg[:, 0:3]).t().long()

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

    all_edgelist = all_edgelist.to(DEVICE)
    all_edgeattr = all_edgeattr.to(DEVICE)

    return all_edgelist, all_edgeattr
