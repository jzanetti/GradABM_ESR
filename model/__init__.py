from torch import device as torch_device

ALL_PARAMS = [
    "r0",
    "mortality_rate",
    "initial_infected_percentage",
    "random_infected_percentgae",
    "exposed_to_infected_time",
    "infected_to_recovered_or_death_time",
    "infection_gamma_shape",
    "infection_gamma_scale",
    "infection_gamma_scaling_factor",
]

STAGE_INDEX = {"susceptible": 0, "exposed": 1, "infected": 2, "recovered_or_death": 3, "death": 4}

DEVICE = torch_device(f"cuda:0")
# DEVICE = torch_device("cpu")

USE_TEMPORAL_PARAMS = True
USE_RNN = False

OPT_METHOD = "sgd"
OPT_METRIC = "mse"
USE_LOSS_SCALER = False
INITIAL_LOSS = 1e10
