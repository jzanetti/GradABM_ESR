import os

import torch
import torch.nn.functional as F
from torchmetrics.regression import (
    MeanAbsolutePercentageError,
    MeanSquaredError,
    PearsonCorrCoef,
)


def postproc(param_model, prediction, y, remove_warm_up: bool = False) -> dict:
    """Warm-up period must be excluded: warm-up period is usually
        equals to infected_to_recovered_or_death_time, we need to remove it since
        it is not generated by the model (e.g., people gradually die over time). In contrast,
        the death after warm-up will suddenly emerge from the initial infection.

    Args:
        param_model (_type_): _description_
        prediction (_type_): _description_
        y (_type_): _description_
    """
    output = {
        "pred": prediction["prediction"][0, :],
        "all_records": prediction["all_records"],
        "y": y[0, :, 0],
    }
    if remove_warm_up:
        infected_to_recovered_or_death_time_index = param_model.learnable_param_order.index(
            "infected_to_recovered_or_death_time"
        )

        exposed_to_infected_time_index = param_model.learnable_param_order.index(
            "exposed_to_infected_time"
        )
        warm_up_period_end = (
            param_model.max_values[infected_to_recovered_or_death_time_index]
            + param_model.max_values[exposed_to_infected_time_index]
        )

        output["pred"] = output["pred"][int(warm_up_period_end.item()) :]

        if output["all_records"] is not None:
            output["all_records"] = output["all_records"][int(warm_up_period_end.item()) :]

        output["y"] = output["y"][int(warm_up_period_end.item()) :]

    return output


def get_loss_func(
    param_model,
    total_timesteps,
    device,
    lr: float = 0.0001,
    weight_decay: float = 0.0,  # 0.01
    opt_method: str = "adam",
    loss_method: str = "mse",  # mse or mspe
):
    """Obtain loss function

    Args:
        param_model (_type_): _description_
        lr (float, optional): Learning rate. Defaults to 0.0001.
        weight_decay (float, optional): Weight decay parameters. Defaults to 0.01.

    Returns:
        _type_: _description_
    """
    if opt_method == "adam":
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, param_model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            differentiable=False,
        )  # loss_fn = NegativeCosineSimilarityLoss()

    elif opt_method == "sgd":
        opt = torch.optim.SGD(
            filter(lambda p: p.requires_grad, param_model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            differentiable=True,
        )

    elif opt_method == "adag":
        opt = torch.optim.Adagrad(
            filter(lambda p: p.requires_grad, param_model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            differentiable=False,
        )

    elif opt_method == "rmsp":
        opt = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, param_model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            differentiable=False,
        )

    elif opt_method == "adad":
        opt = torch.optim.Adadelta(
            filter(lambda p: p.requires_grad, param_model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            differentiable=False,
        )

    loss_weight = torch.ones((1, total_timesteps, 1)).to(device)

    if loss_method == "mse":
        loss_func = MeanSquaredError().to(device)
    elif loss_method == "mspe":
        loss_func = MeanAbsolutePercentageError().to(device)

    return {
        "loss_func": loss_func,
        "opt": opt,
        "loss_weight": loss_weight,
    }


class SeqData(torch.utils.data.Dataset):
    def __init__(self, y):
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.y[idx]


def get_dir_from_path_list(path):
    outdir = path[0]
    if not (os.path.exists(outdir)):
        os.makedirs(outdir)
    for p in path[1:]:
        outdir = os.path.join(outdir, p)
        if not (os.path.exists(outdir)):
            os.makedirs(outdir)
    return outdir


def round_a_list(input: list, sig_figures: int = 3):
    return [round(x, sig_figures) for x in input]
