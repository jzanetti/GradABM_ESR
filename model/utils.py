import os

import torch


def get_loss_func(param_model, lr: float = 0.0001, weight_decay: float = 0.01):
    """Obtain loss function

    Args:
        param_model (_type_): _description_
        lr (float, optional): Learning rate. Defaults to 0.0001.
        weight_decay (float, optional): Weight decay parameters. Defaults to 0.01.

    Returns:
        _type_: _description_
    """
    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, param_model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    return {"loss_func": torch.nn.MSELoss(reduction="none"), "opt": opt}


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
