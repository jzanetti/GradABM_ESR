from copy import copy

import torch
from matplotlib.pyplot import close, plot, savefig

from model.abm import build_simulator, forward_simulator, param_model_forward
from model.inputs import create_agents, read_infection_cfg, train_data_wrapper
from model.interaction import create_interactions
from model.param_model import create_param_model
from model.utils import get_loss_func

device = torch.device("cpu")

y_path = "data/target.csv"
agent_path = "data/agents.csv"
interaction_graph_path = "data/interaction_graph_cfg.csv"

interaction_cfg_path = "cfg/interaction.yml"
infection_cfg_path = "cfg/infection.yml"

print("Step 1: Creating training background data ...")
train_loader = train_data_wrapper(y_path)

print("Step 2: Creating agents ...")
all_agents = create_agents(agent_path)

print("Step 3: Creating interactions ...")
all_interactions = create_interactions(
    interaction_cfg_path, interaction_graph_path, len(all_agents)
)

print("Step 4: Reading infection cfg ...")
infection_cfg = read_infection_cfg(infection_cfg_path)


print("Step 4: Creating initial parameters (to be trained) ...")
param_model = create_param_model(device)


print("Step 5: Creating loss function ...")
loss_def = get_loss_func(param_model, lr=0.001)

num_epochs = 1000
epoch_losses = []
for epi in range(num_epochs):
    epoch_loss = 0
    for batch, y in enumerate(train_loader):
        total_timesteps = y.shape[1]
        param_values = param_model_forward(param_model)

        abm = build_simulator([device], all_agents, all_interactions, infection_cfg)

        predictions = forward_simulator(param_values, abm, total_timesteps, [device])

        loss_weight = torch.ones((1, total_timesteps, 1)).to(device)
        loss = (loss_weight * loss_def["loss_func"](y, predictions)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(param_model.parameters(), 10.0)
        loss_def["opt"].step()
        loss_def["opt"].zero_grad(set_to_none=True)
        epoch_loss += torch.sqrt(loss.detach()).item()

    # print(param_values)
    print(epi, epoch_loss)
    epoch_losses.append(epoch_loss)

print(predictions)
plot(epoch_losses)
print(param_values)
savefig("test.png")
close()
