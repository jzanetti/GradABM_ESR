from itertools import combinations
from pickle import load as pickle_load

from numpy import array
from pandas import DataFrame
from torch import tensor

from process import DEVICE


def load_test_cfg() -> dict:
    return {
        "ensembles": 1,
        "infection": {
            "scaling_factor": {
                "age_dependant": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "symptom_dependant": [0.0, 0.3, 1.0, 0.0, 0.0],
                "ethnicity_dependant": [1.0, 1.0, 10.0, 1.0, 1.1],
                "gender_dependant": [1.0, 1.0],
                "vaccine_dependant": [1.0, 1.0],
            }
        },
        "outbreak_ctl": {
            "isolation": {
                "enable": False,
                "compliance_rate": 0.7,
                "isolation_sf": 0.01,
            },
            "exclusion": {"high_risk_settings": ["school"], "compliance_rate": 1.0},
            "school_closure": {"enable": False, "scaling_factor": 0.1},
        },
        "interaction": {
            "interaction_ratio": 0.5,
            "venues": {
                "school": {"mu": 2.0, "bn": 3.0},
                "household": {"mu": 3.0, "bn": 5.0},
                "travel": {"mu": 1.0, "bn": 1.5},
                "restaurant": {"mu": 1.0, "bn": 1.0},
                "company": {"mu": 1.5, "bn": 0.75},
                "supermarket": {"mu": 0.75, "bn": 0.3},
                "pharmacy": {"mu": 0.75, "bn": 0.5},
            },
        },
        "learnable_params": {
            "vaccine_efficiency_spread": {
                "enable": True,
                "min": 0.0001,
                "max": 0.9,
                "default": 0.1,
            },
            "vaccine_efficiency_symptom": {
                "enable": False,
                "min": 0.001,
                "max": 0.02,
                "default": 0.005,
            },
            "contact_tracing_coverage": {
                "enable": False,
                "min": 0.3,
                "max": 0.7,
                "default": 0.5,
            },
            "initial_infected_percentage": {
                "enable": False,
                "min": 10,
                "max": 70,
                "default": 20,
            },
            "random_infected_percentage": {
                "enable": False,
                "min": 0.05,
                "max": 0.15,
                "default": 0.1,
            },
            "exposed_to_infected_time": {
                "enable": False,
                "min": 0.1,
                "max": 1.5,
                "default": 1.0,
            },
            "infected_to_recovered_or_death_time": {
                "enable": False,
                "min": 1.0,
                "max": 3.0,
                "default": 1.0,
            },
            "infection_gamma_shape": {
                "enable": False,
                "min": 1.0,
                "max": 3.0,
                "default": 1.5,
            },
            "infection_gamma_scale": {
                "enable": False,
                "min": 0.01,
                "max": 30.0,
                "default": 10.0,
            },
            "infection_gamma_scaling_factor": {
                "enable": False,
                "min": 300.0,
                "max": 700.0,
                "default": 500.0,
            },
        },
    }


def load_test_data(large_network: bool = False) -> dict:

    if large_network:
        return pickle_load(open("process/input/testdata/measles_dataset.pickle", "rb"))
    else:

        target = [0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        target = tensor(array(target).reshape(1, len(target), 1)).to(DEVICE)

        total_timesteps = 10

        all_agents = {
            "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "age": [0, 1, 2, 2, 2, 4, 3, 4, 4, 1],
            "gender": [1, 0, 0, 1, 1, 0, 0, 1, 0, 0],
            "ethnicity": [0, 0, 0, 0, 1, 2, 0, 0, 1, 0],
            "area": [9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            "vaccine": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
        all_agents = DataFrame.from_dict(all_agents)

        all_edgelist = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2],
            [5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 4, 5],
        ]
        all_edgeattr = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        ]

    all_edgelist = tensor(array(all_edgelist)).to(DEVICE)
    all_edgeattr = tensor(array(all_edgeattr)).to(DEVICE)

    return {
        "model_inputs": {
            "target": target,
            "total_timesteps": total_timesteps,
            "all_agents": all_agents,
            "all_interactions": {
                "all_edgelist": all_edgelist,
                "all_edgeattr": all_edgeattr,
            },
        },
        "cfg": load_test_cfg(),
    }
