USE_RANDOM_INFECTION_DEFAULT = False

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

STAGE_INDEX = {
    "susceptible": 0,
    "exposed": 1,
    "infected": 2,
    "recovered_or_death": 3,
    "death": 4,
}

REMOVE_WARM_UP_TIMESTEPS = None  # must be a integer or None

SMALL_FIX_VALUE = 1e-9

PRINT_MODEL_INFO = True

PRERUN_CFG = {"params_num": 7, "epochs": 10}

OPTIMIZATION_CFG = {
    "opt_method": "sgd",
    "opt_metric": "mse",
    "use_loss_scaler": False,
    "initial_loss": 1e10,
    "basic_lr": 0.1,
    "num_epochs": 15,
    "clip_grad_norm": 10.0,
    "use_temporal_params": True,
    "adaptive_lr": {"enable": True, "step": 15, "reduction_ratio": 0.9},
}

INITIAL_INFECTION_RATIO = {"timestep_0": 0.3, "timestep_1": 0.7}

PERTURBATE_FLAG_DEFAULT = None

VIS_CFG = {"pred_style": "marker"}  # marker or line or range
