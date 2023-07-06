import torch.nn as nn
from numpy import array
from torch import device as torch_device
from torch import rand as torch_rand
from torch import tensor as torch_tensor

from model.clibnn import CalibNN


def create_param_model(device: torch_device):
    return LearnableParams(device).to(device)


def create_param_model2(
    input_data: dict,
    input_metadata: array,
    device: torch_device,
    training_weeks: int = 0,
    param_dim: int = 3,
    scale_output: str = "abm-covid",
    use_clibnn: bool = False,
):
    """Create parameters model

    Args:
        input_data (dict): Input data
        input_metadata (array): Input metadata, including the area information
        device (torch_device): Device information, e.g., torch.device("cpu")
        training_weeks (int, optional): Training weeks. Defaults to 0.
        param_dim (int, optional): Parameters dimensions. Defaults to 3.
            see tests/model/clibnn.py => MIN_VAL_PARAMS
            (e.g., r0, mortality rate, initial_infections_percentage)
        scale_output (str, optional): Scaling the parameters based on the param types. Defaults to "abm-covid".

    Returns:
        _type_: _description_
    """

    if use_clibnn:
        return CalibNN(
            input_metadata.shape[1],
            input_data["x"].shape[2],
            device,
            training_weeks,
            para_dim=param_dim,
            scale_output=scale_output,
        ).to(device)
    else:
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
