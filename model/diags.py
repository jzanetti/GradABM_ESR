from os import makedirs
from os.path import exists, join
from pickle import dump as pickle_dump
from pickle import load as pickle_load

from matplotlib.pyplot import (
    close,
    legend,
    plot,
    savefig,
    tight_layout,
    title,
    xlabel,
    ylabel,
)
from numpy import array, count_nonzero
from torch import load as torch_load
from torch import save as torch_save

from model import STAGE_INDEX


def save_outputs(param_model, workdir):
    if not exists(workdir):
        makedirs(workdir)

    torch_save(param_model["param_model"], join(workdir, "param_model.model"))
    pickle_dump(param_model["params"], open(join(workdir, "params.p"), "wb"))
    pickle_dump(param_model["output_info"], open(join(workdir, "output_info.p"), "wb"))


def load_outputs(param_path: str, output_info_path: str, param_model_path: str):
    param = pickle_load(open(param_path, "rb"))
    output_info = pickle_load(open(output_info_path, "rb"))
    param_model = torch_load(param_model_path)
    # output = pickle_load(open("output.p", "rb"))
    return {"param": param, "output_info": output_info, "param_model": param_model}


def plot_diags(workdir: str, outputs, epoch_loss_lists, temporal_res, apply_norm: bool = False):
    for i, output in enumerate(outputs):
        my_pred = output["pred"].tolist()

        # ----------------------------
        # Plot agents
        # ----------------------------
        susceptible_counts = count_nonzero(
            output["all_records"] == STAGE_INDEX["susceptible"], axis=1
        )
        exposed_counts = count_nonzero(output["all_records"] == STAGE_INDEX["exposed"], axis=1)
        infected_counts = count_nonzero(output["all_records"] == STAGE_INDEX["infected"], axis=1)
        recovered_or_death_counts = count_nonzero(
            output["all_records"] == STAGE_INDEX["recovered_or_death"], axis=1
        )

        if i == 0:
            plot(susceptible_counts, label="Susceptible", color="c")
            plot(exposed_counts, label="Exposed", color="g")
            plot(infected_counts, label="Infected", color="r")
            plot(recovered_or_death_counts, label="Recovery + Death", color="b")
            plot(my_pred, label="Death", color="m")
        else:
            plot(susceptible_counts, color="c")
            plot(exposed_counts, color="g")
            plot(infected_counts, color="r")
            plot(recovered_or_death_counts, color="b")
            plot(my_pred, color="m")

    xlabel(temporal_res)
    ylabel("Number of agents")
    title("Agent symptom")
    legend()
    tight_layout()
    savefig(join(workdir, "Agents.png"), bbox_inches="tight")
    close()

    # ----------------------------
    # Plot losses
    # ----------------------------
    for epoch_loss_list in epoch_loss_lists:
        plot(epoch_loss_list, "k")
    xlabel("Epoch")
    ylabel("Loss")
    title("Loss")
    tight_layout()
    savefig(join(workdir, "loss.png"), bbox_inches="tight")
    close()

    # ----------------------------
    # Plot Prediction/Truth
    # ----------------------------
    for i, output in enumerate(outputs):
        my_pred = output["pred"].tolist()
        if i == 0:
            my_targ = output["y"].tolist()
            plot(my_targ, color="k", linewidth=2.0, label="Truth")
            plot(my_pred, linewidth=1.0, linestyle="--")
        else:
            plot(my_pred, linewidth=1.0, linestyle="--")

    legend()
    title(f"Prediction ({round(sum(my_pred),2)}) vs Truth ({round(sum(my_targ), 2)})")
    xlabel(f"{temporal_res}s")
    ylabel("Cases")
    tight_layout()
    savefig(join(workdir, "prediction_vs_truth.png"), bbox_inches="tight")
    close()
