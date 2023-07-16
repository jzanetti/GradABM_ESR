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
from numpy import NaN, array, count_nonzero

from model import STAGE_INDEX


def plot_diags(output, epoch_loss_list):
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
    savefig("Agents.png", bbox_inches="tight")
    close()

    # ----------------------------
    # Plot losses
    # ----------------------------
    plot(epoch_loss_list)
    xlabel("Epoch")
    ylabel("Loss")
    title("Loss")
    tight_layout()
    savefig("loss.png", bbox_inches="tight")
    close()

    # ----------------------------
    # Plot Prediction/Truth
    # ----------------------------

    my_targ = output["y"].tolist()
    plot(my_pred, label="Prediction")
    plot(my_targ, label="Truth")
    legend()
    title(f"Prediction ({round(sum(my_pred),2)}) vs Truth ({round(sum(my_targ), 2)})")
    xlabel("Time")
    ylabel("Data")
    tight_layout()
    savefig("prediction_vs_truth.png", bbox_inches="tight")
    close()
