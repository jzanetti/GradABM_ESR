import torch.nn as nn
import torch.nn.init as torch_init
from torch import tensor as torch_tensor
from torch import zeros as torch_zeros
from yaml import safe_load as yaml_load

from model import DEVICE, USE_RNN, USE_TEMPORAL_PARAMS
from utils.utils import read_cfg


def param_model_forward(param_model, target):
    if USE_TEMPORAL_PARAMS:
        param_values_all = param_model.forward(target, DEVICE)
    else:
        param_values_all = param_model.forward()

    return param_values_all


def create_param_model(learnable_param: dict):
    params = get_params(learnable_param)

    if USE_TEMPORAL_PARAMS:
        return TemporalNN(params, DEVICE).to(DEVICE)
    return LearnableParams(params, DEVICE).to(DEVICE)


def get_params(learnable_params: dict) -> dict:
    """Get parmaters information

    Args:
        learnable_param (dict): Learnable parmaters

    Returns:
        dict: The dict contains the decoded params
    """
    learnable_param_order = []
    learnable_param_default = {}
    param_min = []
    param_max = []
    learnable_params_list = []
    for param_name in learnable_params:
        if learnable_params[param_name]["enable"]:
            learnable_param_order.append(param_name)
            param_min.append(learnable_params[param_name]["min"])
            param_max.append(learnable_params[param_name]["max"])
            learnable_params_list.append(
                learnable_params[param_name]["default"] / learnable_params[param_name]["max"]
            )
        else:
            learnable_param_default[param_name] = learnable_params[param_name]["default"]

    return {
        "learnable_param_order": learnable_param_order,
        "learnable_param_default": learnable_param_default,
        "param_min": param_min,
        "param_max": param_max,
        "learnable_params_list": learnable_params_list,
    }


class TemporalNN(nn.Module):
    def __init__(self, params, device, input_size=1, hidden_size=240, num_layers=30):
        super(TemporalNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_rnn = USE_RNN
        self.init_run = True
        if USE_RNN:
            self.temporal_model = nn.RNN(
                input_size,
                hidden_size,
                batch_first=True,
                num_layers=num_layers,
                bidirectional=True,
            )
        else:
            self.temporal_model = nn.LSTM(
                input_size,
                hidden_size,
                batch_first=True,
                num_layers=num_layers,
                bidirectional=True,
            )
        self.learnable_param_order = params["learnable_param_order"]
        self.learnable_param_default = params["learnable_param_default"]

        # Apply Xavier initialization to the RNN weights
        for name, param in self.temporal_model.named_parameters():
            if "weight" in name:
                torch_init.xavier_uniform_(param)

        fc = [
            nn.Linear(in_features=hidden_size * 2, out_features=int(hidden_size)),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(in_features=int(hidden_size / 2), out_features=int(hidden_size / 4)),
            nn.ReLU(),
            nn.Linear(
                in_features=int(hidden_size / 4), out_features=len(self.learnable_param_order)
            ),
        ]
        self.fc = nn.Sequential(*fc)
        self.min_values = torch_tensor(params["param_min"]).to(device)
        self.max_values = torch_tensor(params["param_max"]).to(device)
        self.sigmod = nn.Sigmoid().to(device)

    def scale_param(self, x):
        return self.min_values + (self.max_values - self.min_values) * self.sigmod(x)

    def forward(self, x, device):
        if self.init_run:
            h0 = torch_zeros(2 * self.num_layers, x.shape[0], self.hidden_size).to(device)
            if self.use_rnn:
                hc = h0
            else:
                c0 = torch_zeros(2 * self.num_layers, x.shape[0], self.hidden_size).to(device)
                hc = (h0, c0)
            out, _ = self.temporal_model(torch_tensor(x).float(), hc)
            self.init_run = False
        else:
            out, _ = self.temporal_model(torch_tensor(x).float())
        return self.scale_param(self.fc(out))

    def param_info(self):
        return {
            "learnable_param_order": self.learnable_param_order,
            "learnable_param_default": self.learnable_param_default,
        }


# abm_param_dim,device,scale_output_abm
class LearnableParams(nn.Module):
    """doesn't use data signals"""

    def __init__(self, params, device):
        super().__init__()
        self.learnable_params = nn.Parameter(
            torch_tensor(params["learnable_params_list"], device=device)
        )
        self.min_values = torch_tensor(params["param_min"], device=device)
        self.max_values = torch_tensor(params["param_max"], device=device)
        self.learnable_param_order = params["learnable_param_order"]
        self.learnable_param_default = params["learnable_param_default"]
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
