from copy import copy

import numpy as np
import pandas as pd
import torch

from input9999 import SeqData, create_window_seqs
from model.abm import build_simulator, forward_simulator, param_model_forward
from model.clibnn import CalibNN
from model.create_abm_inputs import (
    create_agents,
    create_interactions,
    create_metadata,
    read_cases,
)
from model.param_model import create_param_model

x_path = "data/data_x_short.csv"
y_path = "data/data_y_short.csv"
abm_cfg_path = "/home/zhangs/Github/GradABM/tests/model/params.yaml"
interaction_graph_cfg_path = "data/interaction_graph_cfg.csv"


print("Reading cases ...")
input_data = read_cases(
    x_path=x_path,
    y_path=y_path,
)

print("Creating areas ...")
input_metadata = create_metadata(all_areas=["test_area"])

print("Creating agents ...")
all_agents = create_agents("data/agents.csv")

print("Creating interactions ...")
all_interactions = create_interactions("cfg/interaction.yml", len(all_agents))

abm_cfg_path = "/home/zhangs/Github/GradABM/tests/model/params.yaml"
interaction_graph_cfg_path = "data/interaction_graph_cfg.csv"

train_dataset = SeqData(input_metadata, input_metadata, input_data["x"], input_data["y"], None)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
device = torch.device("cpu")
# param_model = create_param_model(input_data, input_metadata, device)
param_model = create_param_model(device)

lr = 0.0001

opt = torch.optim.Adam(
    filter(lambda p: p.requires_grad, param_model.parameters()), lr=lr, weight_decay=0.01
)
loss_fcn = torch.nn.MSELoss(reduction="none")

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

training_num_steps = params["num_steps"]  # 70
# training_num_steps = y[0].shape[1]
CLIP = 10
num_epochs = 10
for epi in range(num_epochs):
    epoch_loss = 0
    for batch, (counties, meta, x, y) in enumerate(train_loader):
        # construct abm for each forward pass
        abm = build_simulator(
            copy(params),
            [device],
            counties,
            abm_cfg_path,
            all_agents,
            all_interactions,
            interaction_graph_cfg_path,
        )
        # forward pass param model
        meta = meta.to(device)
        # x = x.to(device)
        # y = y.to(device)

        # param_values = param_model_forward(param_model, params, x, meta)

        param_values = param_model.forward()
        param_values = param_values.repeat((meta.shape[0], 1))

        predictions = forward_simulator(
            params, param_values, abm, training_num_steps, counties, [device]
        )

        loss_weight = torch.ones((len(counties), training_num_steps, 1)).to(device)
        loss = (loss_weight * loss_fcn(y, predictions)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(param_model.parameters(), CLIP)
        opt.step()
        opt.zero_grad(set_to_none=True)
        epoch_loss += torch.sqrt(loss.detach()).item()
