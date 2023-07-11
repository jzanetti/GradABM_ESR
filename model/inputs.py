import torch
from numpy import array
from numpy import float32 as numpy_float32
from numpy import float64 as numpy_float64
from numpy import zeros
from pandas import DataFrame
from pandas import read_csv as pandas_read_csv
from torch import device as torch_device
from torch.utils.data.dataloader import DataLoader
from yaml import safe_load as yaml_load

from model.utils import SeqData


def read_infection_cfg(infection_cfg_path: str):
    """Read infection configuration

    Args:
        infection_cfg_path (str): Infection configuration path
    """
    with open(infection_cfg_path, "r") as fid:
        cfg = yaml_load(fid)

    return cfg


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
    y_data = read_cases(y_path)

    train_dataset = SeqData(y_data)

    return {
        "train_dataset": torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        ),
        "total_timesteps": y_data.shape[1],
    }


def create_metadata(all_areas: list = ["test_area"]) -> array:
    county_idx = {r: i for i, r in enumerate(all_areas)}

    def one_hot(idx, dim=len(county_idx)):
        ans = zeros(dim, dtype="float32")
        ans[idx] = 1.0
        return ans

    metadata = array([one_hot(county_idx[r]) for r in all_areas])

    return metadata


def read_cases(y_path: str) -> dict:
    """Read target cases

    Args:
        y_path (str): Target cases path

    Returns:
        dict: Y data
    """
    y_input = pandas_read_csv(y_path)
    y_input = y_input.to_numpy()
    y = []
    tensor_y = torch.from_numpy(array([y_input]).astype(numpy_float64))
    y.append(tensor_y)
    y_train = torch.cat(y, axis=0)

    return y_train


def create_agents(agent_filepath: str) -> DataFrame:
    """Create agents

    Args:
        agent_filepath (str): Agent data path

    Returns:
        DataFrame: Agents information
    """
    return pandas_read_csv(agent_filepath)
