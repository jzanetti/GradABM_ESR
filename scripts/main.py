import torch
from torch.optim.lr_scheduler import StepLR

from model.abm import build_simulator, forward_simulator, param_model_forward
from model.diags import plot_diags
from model.inputs import create_agents, read_infection_cfg, train_data_wrapper
from model.interaction import create_interactions
from model.param_model import create_param_model
from model.utils import get_loss_func, postproc

torch.autograd.set_detect_anomaly(True)

# -------------------------------
# Parameters:
# -------------------------------
device = torch.device(f"cuda:0")
remove_warm_up = False
use_loss_scaler = False
use_adaptive_lr = False
use_temporal_params = True

num_epochs = 100
learning_rate = 0.01
opt_method = "sgd"
loss_method = "mse"  # mse, mspe

# -------------------------------
# Read data:
# -------------------------------
y_path = "data/exp2/target_test1.csv"
agent_path = "data/exp2/agents.csv"
interaction_graph_path = "data/exp2/interaction_graph_cfg.csv"

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
param_model = create_param_model(
    learnabl_param_cfg_path, device, use_temporal_params=use_temporal_params, use_rnn=False
)

print("Step 5: Creating loss function ...")
loss_def = get_loss_func(
    param_model,
    train_loader["total_timesteps"],
    device,
    lr=learning_rate,
    opt_method=opt_method,
    loss_method=loss_method,
)

print("Step 6: Building ABM ...")
abm = build_simulator([device], all_agents, all_interactions, infection_cfg)

print("Step 7: Getting parameters ...")
param_info = param_model.param_info()

print("Step 8: Adapative learning rate ...")
if use_adaptive_lr:
    lr_scheduler = StepLR(loss_def["opt"], step_size=20, gamma=0.1)

epoch_loss_list = []
param_values_list = []

if use_loss_scaler:
    scaler = torch.cuda.amp.GradScaler()

for epi in range(num_epochs):
    target = train_loader["train_dataset"].dataset.y.to(device)

    if use_temporal_params:
        param_values_all = param_model.forward(target, device)
    else:
        param_values_all = param_model.forward()

    predictions = forward_simulator(
        param_values_all,
        param_info,
        use_temporal_params,
        abm,
        train_loader["total_timesteps"],
        [device],
        save_records=(epi == num_epochs - 1),
    )

    output = postproc(param_model, predictions, target, remove_warm_up)

    loss = loss_def["loss_func"](output["y"], output["pred"])

    if use_loss_scaler:
        scaler.scale(loss).backward()
        scaler.step(loss_def["opt"])
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(param_model.parameters(), 10.0)
        loss_def["opt"].step()

    if use_adaptive_lr:
        lr_scheduler.step()

    loss_def["opt"].zero_grad(set_to_none=True)

    epoch_loss = loss.detach().item()
    current_lr = loss_def["opt"].param_groups[0]["lr"]
    # print(param_values)
    print(f"{epi}: Loss: {round(epoch_loss, 2)}; Lr: {current_lr}")

    epoch_loss_list.append(epoch_loss)
    # param_values_list.append(param_values)

print(param_values_all)
plot_diags(output, epoch_loss_list)


print("done")
