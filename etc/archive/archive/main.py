import os

import torch
from torch.optim.lr_scheduler import StepLR

from model.abm import build_simulator, forward_simulator, param_model_forward
from model.diags import load_outputs, plot_diags, save_outputs
from model.inputs import create_agents, read_infection_cfg, train_data_wrapper
from model.interaction import create_interactions
from model.param_model import create_param_model
from model.utils import get_loss_func, postproc

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

torch.autograd.set_detect_anomaly(True)

# -------------------------------
# Parameters:
# -------------------------------
device = torch.device(f"cuda:0")
remove_warm_up = False
use_loss_scaler = False
use_adaptive_lr = True
use_temporal_params = True
tasks = ["predict"]

num_epochs = 50
learning_rate = 0.1
opt_method = "sgd"
loss_method = "mse"  # mse, mspe, cosine

# -------------------------------
# Read data:
# -------------------------------
# y_path = "data/exp2/target_orig.csv"
# agent_path = "data/exp2/agents.csv"
# interaction_graph_path = "data/exp2/interaction_graph_cfg.csv"

y_path = "data/exp4/targets3.csv"
agent_path = "data/exp4/agents.parquet"
interaction_graph_path = "data/exp4/interaction_graph_cfg.parquet"

# -------------------------------
# Read configuration:
# -------------------------------
interaction_cfg_path = "cfg/interaction.yml"
infection_cfg_path = "cfg/infection.yml"
learnabl_param_cfg_path = "data/exp4/learnable_param.yml"

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

print("Step 5: Building ABM ...")
abm = build_simulator([device], all_agents, all_interactions, infection_cfg)

if "train" in tasks:
    print("Step 6: Creating initial parameters (to be trained) ...")
    param_model = create_param_model(
        learnabl_param_cfg_path, device, use_temporal_params=use_temporal_params, use_rnn=False
    )

    print("Step 7: Getting parameters ...")
    param_info = param_model.param_info()

    print("Step 8: Creating loss function ...")
    loss_def = get_loss_func(
        param_model,
        train_loader["total_timesteps"],
        device,
        lr=learning_rate,
        opt_method=opt_method,
        loss_method=loss_method,
    )

    print("Step 9: Adapative learning rate ...")
    if use_adaptive_lr:
        lr_scheduler = StepLR(loss_def["opt"], step_size=15, gamma=0.9)

    print("Step 10: Prepare training ...")
    epoch_loss_list = []
    param_values_list = []

    if use_loss_scaler:
        scaler = torch.cuda.amp.GradScaler()

    smallest_loss = 1e10
    output_with_smallest_loss = None


for proc_task in tasks:
    if proc_task == "train":
        target = train_loader["train_dataset"].dataset.y.to(device)
        for epi in range(num_epochs):
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
                save_records=False,
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
            print(
                f"{epi}: Loss: {round(epoch_loss, 2)}/{round(smallest_loss, 2)}; Lr: {round(current_lr, 2)}"
            )

            if epoch_loss < smallest_loss:
                output_with_smallest_loss = output
                param_with_smallest_loss = param_values_all
                smallest_loss = epoch_loss

            epoch_loss_list.append(epoch_loss)

        print(param_values_all)

        save_outputs(
            {"params": param_with_smallest_loss, "param_model": param_model},
            {"output": output_with_smallest_loss, "epoch_loss": epoch_loss_list, "target": target},
        )
    else:
        trained_info = load_outputs()
        predictions = forward_simulator(
            trained_info["params"],
            trained_info["model"].param_info(),
            use_temporal_params,
            abm,
            train_loader["total_timesteps"],
            [device],
            save_records=True,
        )
        output = postproc(
            trained_info["model"], predictions, trained_info["output"]["target"], remove_warm_up
        )
        plot_diags(
            output,
            trained_info["output"]["epoch_loss"],
            apply_norm=False,
        )


print("done")
