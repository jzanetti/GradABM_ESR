import torch.nn as nn
from torch import device as torch_device
from torch import rand as torch_rand
from torch import tensor as torch_tensor


def create_param_model(device: torch_device):
    return LearnableParams(device).to(device)


def create_param_model2(
    device: torch_device,
):
    """Create parameters model

    Args:
        device (torch_device): Device information, e.g., torch.device("cpu")

    Returns:
        _type_: _description_
    """
    return LearnableParams(device).to(device)


# abm_param_dim,device,scale_output_abm
class LearnableParams(nn.Module):
    """doesn't use data signals"""

    def __init__(self, device, param_min=[1.0, 0.001, 0.01], param_max=[8.0, 0.02, 1.0]):
        # param: r0, mortality rate, initial_infections_percentage
        super().__init__()
        self.device = device
        self.learnable_params = nn.Parameter(torch_rand(len(param_min), device=self.device))
        self.min_values = torch_tensor(param_min, device=self.device)
        self.max_values = torch_tensor(param_max, device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        out = self.learnable_params
        """ bound output """
        out = self.min_values + (self.max_values - self.min_values) * self.sigmoid(out)
        return out
