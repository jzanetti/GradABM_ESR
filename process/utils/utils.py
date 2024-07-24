from datetime import datetime
from gc import collect
from logging import INFO, Formatter, StreamHandler, basicConfig, getLogger
from os.path import join

from numpy import array, mean
from yaml import safe_load as yaml_load

logger = getLogger()


from torch.cuda import (
    current_device,
    empty_cache,
    get_device_properties,
    memory_allocated,
)


def clear_cuda_memory(print_memory_usage: bool = False):
    device = current_device()
    available_memory_bytes = get_device_properties(
        device
    ).total_memory - memory_allocated(device)
    available_memory = available_memory_bytes / 1024**3

    if print_memory_usage:
        logger.info(f"available_memory: {available_memory} GB")

    # collect()
    empty_cache()
    collect()


def print_prediction(prd, obs):
    rounded_numbers = [round(num) for num in prd.tolist()]
    rounded_numbers2 = [round(num) for num in obs.tolist()]
    logger.info(f"         * predictions: {rounded_numbers}")
    logger.info(f"         * observation: {rounded_numbers2}")


def print_params_increments(param_values_list):
    if len(param_values_list) < 2:
        return
    param_values1 = param_values_list[-2]
    param_values2 = param_values_list[-1]

    if len(param_values1.shape) == 1:
        param_values2 = array(param_values2.tolist())
        param_values1 = array(param_values1.tolist())

    elif len(param_values1.shape) == 3:
        param_values1 = array(param_values1.tolist())[0, :, :]
        param_values2 = array(param_values2.tolist())[0, :, :]

        param_values1 = mean(param_values1, 0)
        param_values2 = mean(param_values2, 0)

    param_incre = (param_values2 - param_values1) / param_values1

    logger.info(f"     * increments: {param_incre}")


def setup_logging(
    workdir: str = "/tmp",
    start_utc: datetime = datetime.utcnow(),
    logger_name: str = "gradabm_esr",
):
    """set up logging system for tasks

    Returns:
        object: a logging object
    """
    formatter = Formatter(
        "%(asctime)s - %(name)s.%(lineno)d - %(levelname)s - %(message)s"
    )
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(formatter)
    logger_path = join(workdir, f"{logger_name}.{start_utc.strftime('%Y%m%d')}")
    basicConfig(filename=logger_path),
    logger = getLogger()
    logger.setLevel(INFO)
    logger.addHandler(ch)

    return logger


def read_cfg(cfg_path: str, key: str = None) -> dict:
    """Read configuration file

    Args:
        cfg_path (str): configuration path

    Returns:
        dict: configuration
    """
    with open(cfg_path, "r") as fid:
        cfg = yaml_load(fid)

    if key is None:
        return cfg

    return cfg[key]


def round_a_list(input: list, sig_figures: int = 3):
    return [round(x, sig_figures) for x in input]
