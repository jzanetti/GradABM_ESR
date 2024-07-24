from numpy import isnan as numpy_isnan
from torch import hstack as torch_hstack
from torch import log as torch_log
from torch import manual_seed
from torch import ones as torch_ones
from torch import tensor as torch_tensor
from torch.nn.functional import gumbel_softmax

from process import DEVICE
from process.model import TORCH_SEED


def apply_gumbel_softmax(prob_yes: torch_tensor, temporal_seed: int = 0):
    """Apply Gumbel Softmaxfunctions

    Args:
        sample_size (int): The size of samples to be applied with the Gumbel Softmax
        prob_yes (torch_tensor): Probability to be used in the Gembel Softmax

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """

    p = torch_hstack((prob_yes, 1 - prob_yes))
    cat_logits = torch_log(p + 1e-9)
    tries = 0

    while True:

        if TORCH_SEED is not None:
            manual_seed(TORCH_SEED + temporal_seed)

        if tries > 5:
            manual_seed(TORCH_SEED + temporal_seed * 2)

        output = gumbel_softmax(logits=cat_logits, tau=1, hard=True, dim=1)[:, 0]

        if not numpy_isnan(output.cpu().clone().detach().numpy()).any():
            break

        tries += 1

        if tries > 10:
            raise Exception("Not able to create initial infected agents ...")

    return output
