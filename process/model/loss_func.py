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

from process import DEVICE
from process.model import OPTIMIZATION_CFG

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
    basic_lr = OPTIMIZATION_CFG["basic_lr"]
    if OPTIMIZATION_CFG["opt_method"] == "adam":
        opt = Adam(
            filter(lambda p: p.requires_grad, param_model.parameters()),
            lr=basic_lr,
            weight_decay=weight_decay,
            differentiable=False,
        )  # loss_fn = NegativeCosineSimilarityLoss()

    elif OPTIMIZATION_CFG["opt_method"] == "sgd":
        opt = SGD(
            filter(lambda p: p.requires_grad, param_model.parameters()),
            lr=basic_lr,
            # weight_decay=weight_decay,
            # momentum=0.3,
            # weight_decay=0.0001,
            differentiable=False,
        )

    elif OPTIMIZATION_CFG["opt_method"] == "adag":
        opt = Adagrad(
            filter(lambda p: p.requires_grad, param_model.parameters()),
            lr=basic_lr,
            weight_decay=weight_decay,
            differentiable=False,
        )

    elif OPTIMIZATION_CFG["opt_method"] == "rmsp":
        opt = RMSprop(
            filter(lambda p: p.requires_grad, param_model.parameters()),
            lr=basic_lr,
            weight_decay=weight_decay,
            differentiable=False,
        )

    elif OPTIMIZATION_CFG["opt_method"] == "adad":
        opt = Adadelta(
            filter(lambda p: p.requires_grad, param_model.parameters()),
            lr=basic_lr,
            weight_decay=weight_decay,
            differentiable=False,
        )

    loss_weight = torch_ones((1, total_timesteps, 1)).to(DEVICE)

    if OPTIMIZATION_CFG["opt_metric"] == "mse":
        loss_func = MeanSquaredError().to(DEVICE)
    elif OPTIMIZATION_CFG["opt_metric"] == "mspe":
        loss_func = MeanAbsolutePercentageError().to(DEVICE)
    elif OPTIMIZATION_CFG["opt_metric"] == "cosine":
        loss_func = CosineSimilarity().to(DEVICE)

    lr_scheduler = None
    if OPTIMIZATION_CFG["adaptive_lr"]["enable"]:
        lr_scheduler = StepLR(
            opt,
            step_size=OPTIMIZATION_CFG["adaptive_lr"]["step"],
            gamma=OPTIMIZATION_CFG["adaptive_lr"]["reduction_ratio"],
        )

    loss_func_scaler = None
    if OPTIMIZATION_CFG["use_loss_scaler"]:
        loss_func_scaler = GradScaler()

    return {
        "loss_func": loss_func,
        "opt": opt,
        "loss_weight": loss_weight,
        "lr_scheduler": lr_scheduler,
        "loss_func_scaler": loss_func_scaler,
    }


def loss_optimization(loss, param_model, loss_def: dict, print_grad: bool = False):
    if loss_def["loss_func_scaler"] is None:
        loss.backward()
        if OPTIMIZATION_CFG["clip_grad_norm"] is not None:
            clip_grad_norm_(
                param_model.parameters(), OPTIMIZATION_CFG["clip_grad_norm"]
            )
        loss_def["opt"].step()
    else:
        loss_def["loss_func_scaler"].scale(loss).backward()
        loss_def["loss_func_scaler"].step(loss_def["opt"])
        loss_def["loss_func_scaler"].update()

    if print_grad:
        for name, param in param_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                logger.info(f"Parameter: {name}, Gradient: {param.grad}")

    if loss_def["lr_scheduler"] is not None:
        loss_def["lr_scheduler"].step()

    loss_def["opt"].zero_grad(set_to_none=True)

    epoch_loss = loss.detach().item()

    return epoch_loss
