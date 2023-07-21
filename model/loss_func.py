from logging import getLogger

from torch import ones as torch_ones
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
from torch.optim.lr_scheduler import StepLR
from torchmetrics.regression import (
    CosineSimilarity,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)

from model import (
    ADAPTIVE_LEARNING_RATE,
    DEVICE,
    LEARNING_RATE,
    OPT_METHOD,
    OPT_METRIC,
    USE_LOSS_SCALER,
)

logger = getLogger()


def get_loss_func(
    param_model,
    total_timesteps,
    weight_decay: float = 0.0,  # 0.01
):
    """Obtain loss function

    Args:
        param_model (_type_): _description_
        lr (float, optional): Learning rate. Defaults to 0.0001.
        weight_decay (float, optional): Weight decay parameters. Defaults to 0.01.

    Returns:
        _type_: _description_
    """
    if OPT_METHOD == "adam":
        opt = Adam(
            filter(lambda p: p.requires_grad, param_model.parameters()),
            lr=LEARNING_RATE,
            weight_decay=weight_decay,
            differentiable=False,
        )  # loss_fn = NegativeCosineSimilarityLoss()

    elif OPT_METHOD == "sgd":
        opt = SGD(
            filter(lambda p: p.requires_grad, param_model.parameters()),
            lr=LEARNING_RATE,
            weight_decay=weight_decay,
            differentiable=True,
        )

    elif OPT_METHOD == "adag":
        opt = Adagrad(
            filter(lambda p: p.requires_grad, param_model.parameters()),
            lr=LEARNING_RATE,
            weight_decay=weight_decay,
            differentiable=False,
        )

    elif OPT_METHOD == "rmsp":
        opt = RMSprop(
            filter(lambda p: p.requires_grad, param_model.parameters()),
            lr=LEARNING_RATE,
            weight_decay=weight_decay,
            differentiable=False,
        )

    elif OPT_METHOD == "adad":
        opt = Adadelta(
            filter(lambda p: p.requires_grad, param_model.parameters()),
            lr=LEARNING_RATE,
            weight_decay=weight_decay,
            differentiable=False,
        )

    loss_weight = torch_ones((1, total_timesteps, 1)).to(DEVICE)

    if OPT_METRIC == "mse":
        loss_func = MeanSquaredError().to(DEVICE)
    elif OPT_METRIC == "mspe":
        loss_func = MeanAbsolutePercentageError().to(DEVICE)
    elif OPT_METRIC == "cosine":
        loss_func = CosineSimilarity().to(DEVICE)

    lr_scheduler = None
    if ADAPTIVE_LEARNING_RATE["enable"]:
        lr_scheduler = StepLR(
            opt,
            step_size=ADAPTIVE_LEARNING_RATE["step"],
            gamma=ADAPTIVE_LEARNING_RATE["reduction_ratio"],
        )

    loss_func_scaler = None
    if USE_LOSS_SCALER:
        loss_func_scaler = GradScaler()

    return {
        "loss_func": loss_func,
        "opt": opt,
        "loss_weight": loss_weight,
        "lr_scheduler": lr_scheduler,
        "loss_func_scaler": loss_func_scaler,
    }


def loss_optimization(loss, param_model, loss_def: dict):
    if loss_def["loss_func_scaler"] is None:
        loss.backward()
        clip_grad_norm_(param_model.parameters(), 10.0)
        loss_def["opt"].step()
    else:
        loss_def["loss_func_scaler"].scale(loss).backward()
        loss_def["loss_func_scaler"].step(loss_def["opt"])
        loss_def["loss_func_scaler"].update()

    if loss_def["lr_scheduler"] is not None:
        loss_def["lr_scheduler"].step()

    loss_def["opt"].zero_grad(set_to_none=True)

    epoch_loss = loss.detach().item()

    return epoch_loss
