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


def plot_diags(workdir: str, output, epoch_loss_list, apply_norm: bool = False):
    my_pred = output["pred"].tolist()

    # ----------------------------
    # Plot agents
    # ----------------------------
    susceptible_counts = count_nonzero(output["all_records"] == STAGE_INDEX["susceptible"], axis=1)
    exposed_counts = count_nonzero(output["all_records"] == STAGE_INDEX["exposed"], axis=1)
    infected_counts = count_nonzero(output["all_records"] == STAGE_INDEX["infected"], axis=1)
    recovered_or_death_counts = count_nonzero(
        output["all_records"] == STAGE_INDEX["recovered_or_death"], axis=1
    )

    plot(susceptible_counts, label="Susceptible")
    plot(exposed_counts, label="Exposed")
    plot(infected_counts, label="Infected")
    plot(recovered_or_death_counts, label="Recovery + Death")
    plot(my_pred, label="Death")
    xlabel("Days")
    ylabel("Number of agents")
    title("Agent symptom")
    legend()
    tight_layout()
    savefig(join(workdir, "Agents.png"), bbox_inches="tight")
    close()

    # ----------------------------
    # Plot losses
    # ----------------------------
    plot(epoch_loss_list)
    xlabel("Epoch")
    ylabel("Loss")
    title("Loss")
    tight_layout()
    savefig(join(workdir, "loss.png"), bbox_inches="tight")
    close()

    # ----------------------------
    # Plot Prediction/Truth
    # ----------------------------
    my_targ = output["y"].tolist()

    if apply_norm:
        my_pred = array(my_pred) / max(my_pred)
        my_targ = array(my_targ) / max(my_targ)
    plot(my_pred, label="Prediction")
    plot(my_targ, label="Truth")
    legend()
    title(f"Prediction ({round(sum(my_pred),2)}) vs Truth ({round(sum(my_targ), 2)})")
    xlabel("Time")
    ylabel("Data")
    tight_layout()
    savefig(join(workdir, "prediction_vs_truth.png"), bbox_inches="tight")
    close()
