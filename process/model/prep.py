import logging
from copy import deepcopy
from glob import glob
from itertools import product
from logging import getLogger
from os import environ as os_environ
from os import makedirs
from os.path import exists, join

from numpy import linspace
from pandas import read_csv as pandas_read_csv
from torch.autograd import set_detect_anomaly
from yaml import dump as yaml_dump
from yaml import safe_load as yaml_safe_load

from process.model import PRERUN_CFG
from process.model.inputs import create_agents, create_interactions, train_data_wrapper
from process.utils.utils import read_cfg

logger = getLogger()


def get_learnable_params_scaler(
    proc_param: str, learnable_params_scaler_update: dict or None
) -> float:
    """Obtain learnable paramters scaler for prediction

    Args:
        proc_param (str): current processing params such as infection_gamma_scaling_factor
        learnable_params_scaler_update (dictorNone): Updated scaler

    Returns:
        float: _description_
    """
    if learnable_params_scaler_update is None:
        proc_scaler_update = 1.0
    else:
        if proc_param in learnable_params_scaler_update:
            proc_scaler_update = learnable_params_scaler_update[proc_param]
        else:
            proc_scaler_update = 1.0

    return proc_scaler_update


def get_all_pred_pahts(workdir: str) -> dict:
    """Get all paths required for prediction

    Args:
        workdir (str): Working directory, e.g., <wordir>/predict

    Raises:
        Exception: Not able to find trained model/agents/interactions

    Returns:
        dict: updated prediction paths
    """

    all_trained_models_dirs = glob(join(workdir, "..", "train", "model", "member_*"))
    all_agents_path = join(workdir, "..", "input", "agents.parquet")
    all_interactions_paths = glob(
        join(workdir, "..", "input", "interaction_graph_cfg_member_*.parquet")
    )

    if len(all_trained_models_dirs) == 0:
        raise Exception(f"Not able to find any trained model for {workdir}")

    if not exists(all_agents_path):
        raise Exception(f"Not able to find any agents for {workdir}")

    if len(all_interactions_paths) == 0:
        raise Exception(f"Not able to find any interactions for {workdir}")

    return {
        "all_trained_models_dirs": all_trained_models_dirs,
        "agents_path": all_agents_path,
        "all_interactions_paths": all_interactions_paths,
        "total_ens": len(all_trained_models_dirs) * len(all_interactions_paths),
    }


def get_train_all_paths(workdir: str) -> dict:
    """Get all paths for agents and interactions

    Args:
        workdir (str): Working directory

    Returns:
        dict: The dict contains all required data for model training
    """

    agents_path = join(workdir, "input", "agents.parquet")
    interaction_paths = glob(
        join(workdir, "input", "interaction_graph_cfg_member_*.parquet")
    )
    target_path = join(workdir, "input", "target.parquet")

    if not exists(agents_path):
        raise Exception(f"Not able to find any agents data in {workdir}/input")

    if len(interaction_paths) == 0:
        raise Exception(f"Not able to find any interaction data in {workdir}/input")

    if not exists(target_path):
        raise Exception(f"Not able to find target data in {workdir}/input")

    return {
        "agents_path": agents_path,
        "interaction_paths": interaction_paths,
        "target_path": target_path,
    }


def prep_wrapper(
    agents_data_path: str,
    interaction_data_path: str,
    target_data_path: str,
    cfg_path: str,
) -> tuple:
    """Preprocessing wrapper

    Args:
        agents_data_path (str): Agents data path
        interaction_data_path (str): Interaction data path
        target_data_path (str): Target data path

    Returns:
        dict: output includes model_inputs and configuration
    """
    logger.info("      * Prep1: Reading configuration ...")
    cfg = read_cfg(cfg_path, key="train")

    logger.info("      * Prep2: Preparing model running environment ...")
    prep_env()

    logger.info("      * Prep3: Getting model input ...")
    model_inputs = prep_model_inputs(
        agents_data_path, interaction_data_path, target_data_path, cfg["interaction"]
    )

    return model_inputs, cfg


def prep_env():
    """Prepare model running environment"""

    os_environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    set_detect_anomaly(True)


def update_params_for_prerun(cfg: dict):
    # cfg["optimization"][
    #    "use_temporal_params"
    # ] = False  # to speed up the training by avoiding temporal params
    cfg["outbreak_ctl"]["school_closure"][
        "enable"
    ] = False  # school closure is usally very slow
    return cfg


def prep_model_inputs(
    agent_path: str, interaction_path: str, target_data_path: str, interaction_cfg: str
) -> dict:
    """Create inputs for the model

    Args:
        agent_path (str): Base agents path
        interaction_path (str): Interaction path
        target_data_path (str): Target data path
        interaction_cfg (str): Interaction configuration

    Returns:
        dict: contains the information:
           - targets to be aimed
           - total timestep to model
           - processed agents
           - processed interactions
    """
    logger.info("       * Step 1: Creating training background data ...")
    if target_data_path is not None:
        target_data = train_data_wrapper(target_data_path)
    else:
        target_data = {"target": None, "total_timesteps": None}

    logger.info("       * Step 2: Creating agents ...")
    all_agents = create_agents(agent_path, max_agents=None)

    logger.info("       * Step 3: Creating interactions ...")
    all_interactions = create_interactions(
        interaction_cfg, interaction_path, len(all_agents["id"].unique())
    )

    return {
        "target": target_data["target"],
        "total_timesteps": target_data["total_timesteps"],
        "all_agents": all_agents,
        "all_interactions": all_interactions,
    }


def get_prerun_params(cfg: dict) -> list:
    prerun_params = cfg["prerun_params"]
    proc_param_values = {}
    for proc_param in prerun_params:
        if not cfg["learnable_params"][proc_param]["enable"]:
            raise Exception(f"The param for {proc_param} is not enabled")

        proc_param_values[proc_param] = linspace(
            cfg["learnable_params"][proc_param]["min"],
            cfg["learnable_params"][proc_param]["max"],
            PRERUN_CFG["params_num"],
        )

    # Get the keys from the dictionary
    keys = list(proc_param_values.keys())

    # Generate all possible combinations of values for each key
    all_params = []
    for values in product(*(proc_param_values[key] for key in keys)):
        combination = {keys[i]: values[i] for i in range(len(keys))}
        all_params.append(combination)

    return all_params


def update_train_cfg_using_prerun(
    updated_cfg_dir: str, base_cfg_path: str, top_params_num: int = 3
) -> list:
    """Create updated cfg for training using the prerun datasets

    Args:
        updated_cfg_dir (str): The directory holds the updated cfg
        base_cfg_path (str): Base configuration file
        top_params_num (int, optional): How many prerun params to be used. Defaults to 3.

    Returns:
        list: The list of updated cfg path
    """

    if not exists(updated_cfg_dir):
        makedirs(updated_cfg_dir)

    prerun_stats_data = pandas_read_csv(
        join(updated_cfg_dir, "..", "prerun", "prerun_stats.csv")
    )
    top_preprun_params = (
        prerun_stats_data.sort_values(by="lost")
        .sort_values(by="lost")
        .head(top_params_num)
    )

    base_cfg = yaml_safe_load(open(base_cfg_path, "rb"))

    updated_cfg_paths = []
    for index, proc_param in top_preprun_params.iterrows():
        updated_cfg = deepcopy(base_cfg)
        for proc_key in base_cfg["train"]["prerun_params"]:
            proc_value = float(proc_param[proc_key])
            updated_cfg["train"]["learnable_params"][proc_key] = {
                "enable": True,
                "min": proc_value * 0.8,
                "max": proc_value * 1.2,
                "default": proc_value,
            }
        updated_cfg_path = join(updated_cfg_dir, f"cfg_from_prerun_{index}.yaml")
        with open(updated_cfg_path, "w") as fid:
            yaml_dump(updated_cfg, fid, default_flow_style=False)

        updated_cfg_paths.append(updated_cfg_path)

    return updated_cfg_paths
