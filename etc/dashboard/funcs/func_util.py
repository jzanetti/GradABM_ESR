from numpy.random import uniform
from pandas import DataFrame


def perturbate_latlon(data: DataFrame, perturbation_range: float = 0.0001):
    for proc_attr in ["latitude", "longitude"]:
        data.loc[data["type"] == "household", proc_attr] = data.loc[
            data["type"] == "household", proc_attr
        ].apply(
            lambda x: x + uniform(-perturbation_range / 5.0, perturbation_range / 5.0)
        )

        data.loc[data["type"] != "household", proc_attr] = data.loc[
            data["type"] != "household", proc_attr
        ].apply(lambda x: x + uniform(-perturbation_range, perturbation_range))

    return data
