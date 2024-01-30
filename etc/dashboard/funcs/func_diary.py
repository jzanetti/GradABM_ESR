from random import choice as random_choice

from pandas import DataFrame
from pandas import merge as pandas_merge


def get_diary_data(agents: DataFrame, address_data: DataFrame, diary_data: DataFrame):
    """Create agent and diary data

    Args:
        syn_data_path (str): _description_
        diary_data_path (str): _description_

    Returns:
        DataFrame: diary data
    """

    def _random_selection(row):
        try:
            items = row.split(",")
        except AttributeError:
            return row
        return random_choice(items)

    for proc_column in ["company", "supermarket", "restaurant", "household", "school"]:
        agents[proc_column] = agents[proc_column].apply(_random_selection)

    address_data = address_data[address_data["name"].isin(agents.values.ravel())]
    diary_data = diary_data[diary_data.index.isin(agents.index)]

    diary_data = diary_data.melt(id_vars="id", var_name="hour", value_name="spec")
    df = pandas_merge(diary_data, agents, on="id", how="left")[
        [
            "spec",
            "hour",
            "household",
            "restaurant",
            "supermarket",
            "school",
            "company",
            "age",
            "gender",
            "ethnicity",
            "id",
        ]
    ]
    df = df[df["spec"] != "travel"]
    df["name"] = df.apply(lambda row: row[row["spec"]], axis=1)
    df = df.rename(columns={"spec": "type"})
    df = df[["type", "name", "hour", "age", "gender", "ethnicity", "id"]]

    address_data = address_data[
        ["type", "name", "latitude", "longitude"]
    ].drop_duplicates()

    return pandas_merge(
        df[["type", "name", "hour", "age", "gender", "ethnicity", "id"]],
        address_data[["type", "name", "latitude", "longitude"]],
        on=["type", "name"],
        how="left",
    ).dropna()
