from copy import copy

import numpy as np
import pandas as pd
import torch

from model.abm import GradABM, build_simulator, forward_simulator, param_model_forward
from model.clibnn import CalibNN
from model.create_abm_inputs import (
    create_agents,
    create_metadata,
    read_cases,
    train_data_wrapper,
)
from model.interaction import create_interactions
from model.param_model import create_param_model
from model.utils import get_loss_func

device = torch.device("cpu")

y_path = "data/data_y_short.csv"
agent_path = "data/agents.csv"
abm_cfg_path = "/home/zhangs/Github/GradABM/tests/model/params.yaml"
interaction_cfg_path = "cfg/interaction.yml"
interaction_graph_path = "data/interaction_graph_cfg.csv"

print("Step 1: Creating training background data ...")
train_loader = train_data_wrapper(y_path)

print("Step 2: Creating agents ...")
all_agents = create_agents(agent_path)

print("Step 3: Creating interactions ...")
all_interactions = create_interactions(
    interaction_cfg_path, interaction_graph_path, len(all_agents)
)

print("Step 4: Creating initial parameters (to be trained) ...")
param_model = create_param_model(device)

print("Step 5: Creating loss function ...")
loss_def = get_loss_func(param_model)

"""
params = {
    "seed": 6666,
    "num_runs": 1,
    "disease": "COVID",
    "pred_week": "202021",
    "joint": False,
    "inference_only": False,
    "noise_level": 0,
    "state": "MA",
    "county_id": "25001",
    "model_name": "GradABM-learnable-params",  # "GradABM-time-varying", GradABM-learnable-params
    "num_steps": 10,
}
"""
params = {}

CLIP = 10
num_epochs = 10
training_num_steps = 10
for epi in range(num_epochs):
    epoch_loss = 0
    for batch, y in enumerate(train_loader):
        # construct abm for each forward pass
        abm = build_simulator(copy(params), [device], abm_cfg_path, all_agents, all_interactions)

        param_values = param_model_forward(param_model)

        predictions = forward_simulator(param_values, abm, training_num_steps, [device])

        loss_weight = torch.ones((1, training_num_steps, 1)).to(device)
        loss = (loss_weight * loss_def["loss_func"](y, predictions)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(param_model.parameters(), CLIP)
        loss_def["opt"].step()
        loss_def["opt"].zero_grad(set_to_none=True)
        epoch_loss += torch.sqrt(loss.detach()).item()
