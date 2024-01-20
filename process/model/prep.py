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
    logger.info("Prep1: Reading configuration ...")
    cfg = read_cfg(cfg_path, key="train")

    logger.info("Prep2: Preparing model running environment ...")
    prep_env()

    logger.info("Prep3: Getting model input ...")
    model_inputs = prep_model_inputs(
        agents_data_path,
        interaction_data_path,
        target_data_path,
        cfg["interaction"],
        cfg["target"],
        cfg["interaction_ratio"],
    )

    return model_inputs, cfg

    logger.info("Prep4: Building ABM ...")
    abm = build_abm(
        model_inputs["all_agents"],
        model_inputs["all_interactions"],
        cfg["infection"],
        None,
    )

    logger.info("Creating initial parameters (to be trained) ...")

    param_model = create_param_model(
        obtain_param_cfg(cfg["learnable_params"], prerun_params),
        cfg["optimization"]["use_temporal_params"],
    )

    logger.info("Creating loss function ...")
    loss_def = get_loss_func(
        param_model, model_inputs["total_timesteps"], cfg["optimization"]
    )
    epoch_loss_list = []
    param_values_list = []
    smallest_loss = INITIAL_LOSS

    if prerun_params:
        num_epochs = PRERUN_NUM_EPOCHS
        cfg = update_params_for_prerun(cfg)
    else:
        num_epochs = cfg["optimization"]["num_epochs"]


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
