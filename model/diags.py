from matplotlib.pyplot import (
    close,
    legend,
    pcolor,
    plot,
    savefig,
    tight_layout,
    xlabel,
    ylabel,
)
from numpy import array, count_nonzero

from model import STAGE_INDEX


def plot_diags(predictions, epoch_loss_list):
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
    tight_layout()
    savefig("loss.png", bbox_inches="tight")
    close()
