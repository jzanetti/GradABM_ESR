from os import environ as os_environ

from torch.autograd import set_detect_anomaly

from model.inputs import create_agents, create_interactions, train_data_wrapper


def prep_env():
    """Prepare model running environment"""

    os_environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    set_detect_anomaly(True)


def prep_model_inputs(
    agent_path: str, interaction_path: str, target_data_path: str, interaction_cfg: str
) -> dict:
    print("Step 1: Creating training background data ...")
    target_data = train_data_wrapper(target_data_path)

    print("Step 2: Creating agents ...")
    all_agents = create_agents(agent_path)

    print("Step 3: Creating interactions ...")
    all_interactions = create_interactions(interaction_cfg, interaction_path, len(all_agents))

    return {
        "target": target_data["target"],
        "total_timesteps": target_data["total_timesteps"],
        "all_agents": all_agents,
        "all_interactions": all_interactions,
    }
