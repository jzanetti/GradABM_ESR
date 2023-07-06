from copy import copy

import torch
from matplotlib.pyplot import close, plot, savefig

from model.abm import build_simulator, forward_simulator, param_model_forward
from model.inputs import create_agents, train_data_wrapper
from model.interaction import create_interactions
from model.param_model import create_param_model
from model.utils import get_loss_func

device = torch.device("cpu")

y_path = "data/target.csv"
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

num_epochs = 3000
epoch_losses = []
for epi in range(num_epochs):
    epoch_loss = 0
    for batch, y in enumerate(train_loader):
        total_timesteps = y.shape[1]

        # construct abm for each forward pass
        abm = build_simulator([device], all_agents, all_interactions)

        param_values = param_model_forward(param_model)

        predictions = forward_simulator(param_values, abm, total_timesteps, [device])

        loss_weight = torch.ones((1, total_timesteps, 1)).to(device)
        loss = (loss_weight * loss_def["loss_func"](y, predictions)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(param_model.parameters(), 10.0)
        loss_def["opt"].step()
        loss_def["opt"].zero_grad(set_to_none=True)
        epoch_loss += torch.sqrt(loss.detach()).item()

    # print(predictions)
    print(epi, param_model.learnable_params)

    epoch_losses.append(epoch_loss)
    print(predictions)

plot(epoch_losses)
savefig("test.png")
close()
