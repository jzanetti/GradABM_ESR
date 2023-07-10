import torch.nn as nn
from torch import device as torch_device
from torch import rand as torch_rand
from torch import tensor as torch_tensor


def create_param_model(device: torch_device):
    return LearnableParams(device).to(device)


# abm_param_dim,device,scale_output_abm
class LearnableParams(nn.Module):
    """doesn't use data signals"""

    def __init__(
        self,
        device,
        # param_min=[1.0, 0.01, 0.0, 0.0, 1.0, 0.0, 0.1, 0.1],
        # param_max=[100.0, 10.0, 50.0, 10.0, 15.0, 10.0, 1.5, 15.0],
        param_min=[1.0, 0.01, 0.0, 0.0, 1.0],
        param_max=[100.0, 10.0, 50.0, 10.0, 15.0],
        num_vars=5,
    ):
        # param:
        # - r0 (overall infection rate): 0
        # - mortality rate: 1
        # - initial_infections_percentage: 2
        # - exposed_to_infected_time: 3
        # - infected_to_recovered_time: 4
        # - infection gamma PDF loc: 5
        # - infection gamma PDF a: 6
        # - infection gamma PDF scale: 7

        super().__init__()
        self.device = device
        self.learnable_params = nn.Parameter(torch_rand(num_vars, device=self.device))
        self.min_values = torch_tensor(param_min, device=self.device)
        self.max_values = torch_tensor(param_max, device=self.device)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        out = self.learnable_params
        out = self.min_values + (self.max_values - self.min_values) * self.sigmoid(out)
        return out
