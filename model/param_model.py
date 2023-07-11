import torch.nn as nn
from torch import device as torch_device
from torch import rand as torch_rand
from torch import tensor as torch_tensor
from yaml import safe_load as yaml_load


def create_param_model(learnable_param_cfg_path: str, device: torch_device):
    return LearnableParams(learnable_param_cfg_path, device).to(device)


# abm_param_dim,device,scale_output_abm
class LearnableParams(nn.Module):
    """doesn't use data signals"""

    def __init__(self, learnable_param_cfg_path, device):
        # param order example:
        # - r0 (overall infection rate): 0
        # - mortality rate: 1
        # - initial_infections_percentage: 2
        # - exposed_to_infected_time: 3
        # - infected_to_recovered_time: 4
        # - infection gamma PDF loc: 5
        # - infection gamma PDF a: 6
        # - infection gamma PDF scale: 7

        with open(learnable_param_cfg_path, "r") as fid:
            learnable_param = yaml_load(fid)

        learnable_param_order = []
        learnable_param_default = {}
        param_min = []
        param_max = []
        for param_name in learnable_param["learnable_params"]:
            if learnable_param["learnable_params"][param_name]["enable"]:
                learnable_param_order.append(param_name)
                param_min.append(learnable_param["learnable_params"][param_name]["min"])
                param_max.append(learnable_param["learnable_params"][param_name]["max"])
            else:
                learnable_param_default[param_name] = learnable_param["learnable_params"][
                    param_name
                ]["default"]

        super().__init__()
        # self.device = device
        self.learnable_params = nn.Parameter(torch_rand(len(learnable_param_order), device=device))
        self.min_values = torch_tensor(param_min, device=device)
        self.max_values = torch_tensor(param_max, device=device)
        self.learnable_param_order = learnable_param_order
        self.learnable_param_default = learnable_param_default
        self.scaling_func = nn.Sigmoid()

    def forward(self):
        return self.min_values + (self.max_values - self.min_values) * self.scaling_func(
            self.learnable_params
        )

    def param_info(self):
        return {
            "learnable_param_order": self.learnable_param_order,
            "learnable_param_default": self.learnable_param_default,
        }
