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

from model import STAGE_INDEX


def plot_diags(predictions, target, epoch_loss_list):
    # ----------------------------
    # Plot agents
    # ----------------------------
    exposed_counts = count_nonzero(predictions["all_records"] == STAGE_INDEX["exposed"], axis=1)
    infected_counts = count_nonzero(predictions["all_records"] == STAGE_INDEX["infected"], axis=1)
    recovered_or_death_counts = count_nonzero(
        predictions["all_records"] == STAGE_INDEX["recovered_or_death"], axis=1
    )
    plot(exposed_counts, label="Exposed")
    plot(infected_counts, label="Infected")
    plot(recovered_or_death_counts, label="Recovered/Death")
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
    my_pred = [item for sublist in predictions["prediction"][0, :, :].tolist() for item in sublist]
    my_targ = target[0, :, 0].tolist()
    plot(my_pred, label="Prediction")
    plot(my_targ, label="Truth")
    legend()
    title(f"Prediction ({sum(my_pred)}) vs Truth ({sum(my_pred)})")
    xlabel("Time")
    ylabel("Data")
    tight_layout()
    savefig("prediction_vs_truth.png", bbox_inches="tight")
    close()
