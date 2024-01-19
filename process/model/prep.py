import logging
from itertools import product
from logging import getLogger
from os import environ as os_environ

from numpy import linspace
from torch.autograd import set_detect_anomaly

from process.model import PRERUN_PARAMS_NUM
from process.model.inputs import create_agents, create_interactions, train_data_wrapper
from utils.utils import read_cfg

logger = getLogger()


def prep_env():
    """Prepare model running environment"""

    os_environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    set_detect_anomaly(True)


def update_params_for_prerun(cfg: dict):
    # cfg["optimization"][
    #    "use_temporal_params"
    # ] = False  # to speed up the training by avoiding temporal params
    cfg["infection"]["outbreak_ctl"]["school_closure"][
        "enable"
    ] = False  # school closure is usally very slow
    return cfg


def prep_model_inputs(
    agent_path: str,
    interaction_path: str,
    target_data_path: str,
    interaction_cfg: str,
    target_cfg: dict,
    interaction_ratio_cfg: float,
) -> dict:
    """Create inputs for the model

    Args:
        agent_path (str): Base agents path
        interaction_path (str): Interaction path
        target_data_path (str): Target data path
        interaction_cfg (str): Interaction configuration
        target_cfg (dict): Target confiration
        interaction_ratio_cfg (float): How many interaction to be drawn

    Returns:
        dict: contains the information:
           - targets to be aimed
           - total timestep to model
           - processed agents
           - processed interactions
    """
    logger.info("Step 1: Creating training background data ...")
    target_data = train_data_wrapper(target_data_path, target_cfg)

    logger.info("Step 2: Creating agents ...")
    all_agents = create_agents(agent_path, max_agents=None)

    logger.info("Step 3: Creating interactions ...")
    all_interactions = create_interactions(
        interaction_cfg,
        interaction_path,
        len(all_agents["id"].unique()),
        interaction_ratio_cfg,
    )

    # all_agents2 = all_agents.reset_index()
    # all_agents2["index"] = all_agents2.index
    # index_id_map = dict(zip(all_agents2['id'], all_agents2['index']))

    return {
        "target": target_data["target"],
        "total_timesteps": target_data["total_timesteps"],
        "all_agents": all_agents,
        "all_interactions": all_interactions,
    }


def get_prerun_params(model_cfg_path: str) -> list:
    cfg = read_cfg(model_cfg_path, key="train")
    prerun_params = cfg["prerun_params"]
    proc_param_values = {}
    for proc_param in prerun_params:
        if not cfg["learnable_params"][proc_param]["enable"]:
            raise Exception(f"The param for {proc_param} is not enabled")

        proc_param_values[proc_param] = linspace(
            cfg["learnable_params"][proc_param]["min"],
            cfg["learnable_params"][proc_param]["max"],
            PRERUN_PARAMS_NUM,
        )

    # Get the keys from the dictionary
    keys = list(proc_param_values.keys())

    # Generate all possible combinations of values for each key
    all_params = []
    for values in product(*(proc_param_values[key] for key in keys)):
        combination = {keys[i]: values[i] for i in range(len(keys))}
        all_params.append(combination)

    return all_params
