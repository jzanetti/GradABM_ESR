from copy import copy

import torch
from matplotlib.pyplot import close, plot, savefig

from model.abm import build_simulator, forward_simulator, param_model_forward
from model.diags import plot_diags
from model.inputs import create_agents, read_infection_cfg, train_data_wrapper
from model.interaction import create_interactions
from model.param_model import create_param_model
from model.utils import get_loss_func

# device = torch.device("cpu")
device = torch.device(f"cuda:0")
# -------------------------------
# Read data:
# -------------------------------
y_path = "data/exp1/target.csv"
agent_path = "data/exp1/agents.csv"
interaction_graph_path = "data/exp1/interaction_graph_cfg.csv"

# -------------------------------
# Read configuration:
# -------------------------------
interaction_cfg_path = "cfg/interaction.yml"
infection_cfg_path = "cfg/infection.yml"
learnabl_param_cfg_path = "cfg/learnable_param.yml"

# -------------------------------
# Program starts from here:
# -------------------------------
print("Step 1: Creating training background data ...")
train_loader = train_data_wrapper(y_path)

print("Step 2: Creating agents ...")
all_agents = create_agents(agent_path)

print("Step 3: Creating interactions ...")
all_interactions = create_interactions(
    interaction_cfg_path, interaction_graph_path, len(all_agents), device
)

print("Step 4: Reading infection cfg ...")
infection_cfg = read_infection_cfg(infection_cfg_path)


print("Step 4: Creating initial parameters (to be trained) ...")
param_model = create_param_model(learnabl_param_cfg_path, device)


print("Step 5: Creating loss function ...")
loss_def = get_loss_func(
    param_model, train_loader["total_timesteps"], device, lr=1e-2, opt_method="adam"
)  # adam or sgd

print("Step 6: Building ABM ...")
abm = build_simulator([device], all_agents, all_interactions, infection_cfg)

print("Step 7: Getting parameters ...")
param_info = param_model.param_info()

num_epochs = 1000
epoch_loss_list = []
param_values_list = []

for epi in range(num_epochs):
    epoch_loss = 0
    for batch, y in enumerate(train_loader["train_dataset"]):
        y = y.to(device)
        total_timesteps = y.shape[1]
        param_values = param_model_forward(param_model)

        save_record = False
        if epi == num_epochs - 1:
            save_record = True

        predictions = forward_simulator(
            param_values,
            param_info,
            abm,
            train_loader["total_timesteps"],
            [device],
            save_record=save_record,
        )

        loss = (
            loss_def["loss_weight"] * loss_def["loss_func"](y, predictions["prediction"])
        ).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(param_model.parameters(), 10.0)
        loss_def["opt"].step()
        loss_def["opt"].zero_grad(set_to_none=True)
        epoch_loss += torch.sqrt(loss.detach()).item()

    # print(param_values)
    print(f"{epi}: {epoch_loss}, {param_values}")

    epoch_loss_list.append(epoch_loss)
    param_values_list.append(param_values)


plot_diags(predictions, epoch_loss_list)

savefig("test.png")
close()
