ALL_PARAMS = [
    "r0",
    "mortality_rate",
    "initial_infected_percentage",
    "exposed_to_infected_time",
    "infected_to_recovered_or_death_time",
    "infection_gamma_shape",
    "infection_gamma_scale",
]

STAGE_INDEX = {"susceptible": 0, "exposed": 1, "infected": 2, "recovered_or_death": 3}


BEST_LOSS = float("inf")
