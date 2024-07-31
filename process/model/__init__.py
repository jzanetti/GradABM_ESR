USE_RANDOM_EXPOSED_DEFAULT = True

USE_RNN = False

ALL_PARAMS = [
    "vaccine_efficiency_spread",
    "vaccine_efficiency_symptom",
    "contact_tracing_coverage",
    "initial_infected_percentage",
    "random_infected_percentage",
    "exposed_to_infected_time",
    "infected_to_recovered_or_death_time",
    "infection_gamma_shape",
    "infection_gamma_scale",
    "infection_gamma_scaling_factor",
]

"""
STAGE_INDEX = {
    "susceptible": 0,
    "exposed": 1,
    "infected": 2,
    "recovered_or_death": 3,
    "death": 4,
}
"""

# SUSCEPTIBLE_STAGE_INDEX = 0
STAGE_INDEX = {
    "susceptible": 1,
    "exposed": 2,
    "infected": 4,
    "recovered_or_death": 8,
    "infected_target": 999,
}

SMALL_FIX_VALUE = 1e-9

PRINT_MODEL_INFO = False

PRINT_MODEL_INFO2 = False

PRERUN_CFG = {"params_num": 7, "epochs": 10}

# ----------------------------------------
# when use_temporal_params is True, it is recommended to use:
# - opt_method: sgd, basic_lr: 0.01, clip_grad_norm: 10.0
# otherwise:
# - opt_method: adam, basic_lr: 0.1, clip_grad_norm: None
# ----------------------------------------
OPTIMIZATION_CFG = {
    "opt_method": "sgd",
    "opt_metric": "mse",
    "use_loss_scaler": False,
    "initial_loss": 1e10,
    "basic_lr": 0.01,
    "num_epochs": 100,
    "clip_grad_norm": 10.0,
    "use_temporal_params": True,
    "adaptive_lr": {"enable": False, "step": 15, "reduction_ratio": 0.9},
    "scaler": 1.0,
    "warmup_timestep": 2,
}

PERTURBATE_FLAG_DEFAULT = None

VIS_CFG = {"pred_style": "marker"}  # marker or line or range

TORCH_SEED = 123

MISSING_DATA = -99
