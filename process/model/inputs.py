import torch
from numpy import array
from numpy import float64 as numpy_float64
from pandas import DataFrame
from pandas import cut as pandas_cut
from pandas import read_csv as pandas_read_csv
from pandas import read_parquet as pandas_read_parquet
from pandas import to_numeric as pandas_to_numeric
from torch import hstack as torch_hstack
from torch import ones as torch_ones
from torch import split as torch_split
from torch import tensor as torch_tensor
from torch import vstack as torch_vstack
from torch.utils.data.dataloader import DataLoader

from process import (
    AGE_INDEX,
    DEVICE,
    ETHNICITY_INDEX,
    GENDER_INDEX,
    LOC_INDEX,
    VACCINE_INDEX,
)


class SeqData(torch.utils.data.Dataset):
    def __init__(self, y):
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.y[idx]


def train_data_wrapper(
    y_path,
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
        y_input = pandas_read_parquet(y_path)
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

    return agent_info_transformation(all_agents)


def agent_info_transformation(agents_data: DataFrame) -> DataFrame:
    """Convert agent information from description to number, e.g., female to 0

    Args:
        agents_data (DataFrame): Original agent data

    Returns:
        DataFrame: updated agent data
    """

    agents = agents_data[
        ["id", "age", "gender", "ethnicity", "area", "vaccine"]
    ].drop_duplicates()

    age_bins = [int(k.split("-")[0]) for k in AGE_INDEX.keys()] + [999]
    age_labels = list(AGE_INDEX.values())
    agents["age"] = pandas_cut(
        agents["age"], bins=age_bins, labels=age_labels, include_lowest=True
    )
    agents["gender"] = agents["gender"].map(GENDER_INDEX)
    agents["vaccine"] = agents["vaccine"].map(VACCINE_INDEX)
    agents["ethnicity"] = agents["ethnicity"].map(ETHNICITY_INDEX)

    # agents = agents.reset_index()

    # agents = agents.drop(["index"], axis=1, inplace=False)

    # agents["id"] = agents.index

    return agents


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
    agents_mean_interactions_output = {}
    agents_mean_interactions_split_output = {}

    for proc_key in ["mu", "bn"]:
        agents_mean_interactions_output[proc_key] = 0 * torch_ones(
            num_agents, len(LOC_INDEX)
        ).to(DEVICE)
        for network_index in network_type_dict_inv:
            proc_interaction = (
                torch_tensor(
                    agents_mean_interaction_cfg[network_type_dict_inv[network_index]][
                        proc_key
                    ]
                )
                .float()
                .to(DEVICE)
            )

            agents_mean_interactions_output[proc_key][
                :, network_index
            ] = proc_interaction

        agents_mean_interactions_split_output[proc_key] = list(
            torch_split(agents_mean_interactions_output[proc_key], 1, dim=1)
        )

        agents_mean_interactions_split_output[proc_key] = [
            a.view(-1) for a in agents_mean_interactions_split_output[proc_key]
        ]

    return {
        "agents_mean_interactions_mu": agents_mean_interactions_output["mu"],
        "agents_mean_interactions_mu_split": agents_mean_interactions_split_output[
            "mu"
        ],
        "agents_mean_interactions_bn": agents_mean_interactions_output["bn"],
        "agents_mean_interactions_bn_split": agents_mean_interactions_split_output[
            "bn"
        ],
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

    # ----------------------------
    # Get edges interaction intensity and frequency
    # ----------------------------
    if interaction_graph_path.endswith("csv"):
        create_bidirection = True
        edges_mean_interaction_cfg = pandas_read_csv(
            interaction_graph_path, header=None
        )
    if interaction_graph_path.endswith("parquet"):
        create_bidirection = False
        edges_mean_interaction_cfg = pandas_read_parquet(interaction_graph_path)

    edges_mean_interaction_cfg = edges_mean_interaction_cfg.sample(
        frac=interaction_cfg["interaction_ratio"]
    )

    agents_mean_interactions = {"bn": {}, "mu": {}}
    for network_name in network_type_dict:
        for proc_interaction_key in list(agents_mean_interactions.keys()):
            agents_mean_interactions[proc_interaction_key][
                network_type_dict[network_name]
            ] = interaction_cfg["venues"][network_name][proc_interaction_key]

    all_edgelist, all_edgeattr = init_interaction_graph(
        edges_mean_interaction_cfg, agents_mean_interactions, create_bidirection
    )

    return {
        "all_edgelist": all_edgelist,
        "all_edgeattr": all_edgeattr,
    }


def init_interaction_graph(
    interaction_graph_cfg: dict,
    agents_mean_interactions: dict,
    create_bidirection: bool,
) -> tuple:
    """Initialize interaction graph

    Args:
        interaction_graph_path (str): Interaction graph path
        network_type_dict (dict): Network setup
        agents_mean_interactions_bn (dict): The scale-factor for the network on which the interaction occured
        device (_type_): Device type, e.g., cpu

    Returns:
        all_edgelist [shape: 2 * interactions]:
            The edge list for showing the link between nodes. E.g.,
                tensor([[1495108, 1227628, 1451663,  ..., 1292632, 1237066, 1251684],
                        [1497264, 1228306, 1450580,  ..., 1293318, 1234397, 1242145]])
        all_edgeattr: [shape: 2 * interactions]:
            The edge attributes: the first row: venue type (e.g., household, school etc.)
                                 the second row: venue contact intensity.
                tensor([[5., 5., 0.,  ..., 5., 0., 3.],
                        [1., 1., 5.,  ..., 1., 5., 3.]])
    """
    interaction_graph_cfg = (
        interaction_graph_cfg[["id_x", "id_y", "spec"]]
        .apply(pandas_to_numeric, errors="coerce")
        .values.astype(int)
    )

    if create_bidirection:
        random_network_edgelist_forward = (
            torch_tensor(interaction_graph_cfg[:, 0:3]).t().long()
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
    else:
        random_network_edge_all = torch_tensor(interaction_graph_cfg[:, 0:4]).t().long()

    random_network_edgelist = random_network_edge_all[0:2, :]

    random_network_edgeattr_type = random_network_edge_all[2, :]

    random_network_edgeattr_value = {}
    for edgeattr_key in ["mu", "bn"]:
        random_network_edgeattr_value[edgeattr_key] = [
            agents_mean_interactions[edgeattr_key][key]
            for key in random_network_edgeattr_type.tolist()
        ]

    random_network_edgeattr = torch_vstack(
        (
            random_network_edgeattr_type,
            torch_tensor(random_network_edgeattr_value["mu"]),
            torch_tensor(random_network_edgeattr_value["bn"]),
        )
    )

    all_edgelist = torch_hstack((random_network_edgelist,))
    all_edgeattr = torch_hstack((random_network_edgeattr,))

    all_edgelist = all_edgelist.to(DEVICE)
    all_edgeattr = all_edgeattr.to(DEVICE)

    return all_edgelist, all_edgeattr
