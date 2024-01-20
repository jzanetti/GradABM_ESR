# import os

from torch import device as torch_device

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "500"

ALL_PARAMS = [
    "vaccine_efficiency_spread",
    "vaccine_efficiency_symptom",
    "contact_tracing_coverage",
    "initial_infected_percentage",
    "random_infected_percentgae",
    "exposed_to_infected_time",
    "infected_to_recovered_or_death_time",
    "infection_gamma_shape",
    "infection_gamma_scale",
    "infection_gamma_scaling_factor",
]

STAGE_INDEX = {
    "susceptible": 0,
    "exposed": 1,
    "infected": 2,
    "recovered_or_death": 3,
    "death": 4,
}

DEVICE = torch_device(f"cuda:0")
# DEVICE = torch_device("cpu")

"""
TORCH_SEED_NUM = {
    "initial_infected": 100,
    "random_infected": 200,
    "newly_exposed": 300,
    "isolation_policy": 400,
}
"""


TORCH_SEED_NUM = None

USE_RNN = False
REMOVE_WARM_UP_TIMESTEPS = None  # must be a integer or None

OPT_METHOD = "sgd"
OPT_METRIC = "mse"
USE_LOSS_SCALER = False
INITIAL_LOSS = 1e10

SMALL_VALUE = 1e-9

PRINT_INCRE = True

PRERUN_PARAMS_NUM = 8
PRERUN_NUM_EPOCHS = 10

OPTIMIZATION_CFG = {
    "basic_lr": 0.1,
    "num_epochs": 30,
    "clip_grad_norm": 10.0,
    "use_temporal_params": True,
    "adaptive_lr": {"enable": True, "step": 15, "reduction_ratio": 0.9},
}
