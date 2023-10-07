from os.path import join

from pandas import DataFrame, concat
from pandas import read_parquet as pandas_read_parquet

from input import AGE_INDEX, ETHNICITY_INDEX, LOC_INDEX, SEX_INDEX


def obtain_agents_info(agents_data_path: str, interaction_data_path: str) -> dict:
    """Obtain agents data for filtering purpose

    Args:
        agents_data_path (str): Agent data path
        interaction_data_path (str): Interaction data path

    Returns:
        dict: processed data
    """

    def _map_age_index(age_index):
        for key, value in AGE_INDEX.items():
            if int(age_index) == value:
                return key
        return None

    agents_data = pandas_read_parquet(agents_data_path)
    interaction_data = pandas_read_parquet(interaction_data_path)
    # agents_data2["age_range"] = agents_data2["age"].apply(lambda x: AGE_INDEX[x])
    agents_data["age_range"] = agents_data["age"].apply(_map_age_index)
    agents_data["age_range_min"] = agents_data["age_range"].str.split("-").str[0]
    agents_data["age_range_max"] = agents_data["age_range"].str.split("-").str[1]

    return {"agents_data": agents_data, "interaction_data": interaction_data}


def age_filter(
    data_to_process: DataFrame, max_age: int or None, min_age: int or None
) -> DataFrame:
    """Age filter

    Args:
        data_to_process (DataFrame): _description_
        max_age (intorNone): _description_
        min_age (intorNone): _description_

    Returns:
        DataFrame: _description_
    """
    if max_age is None:
        max_age = 99999.0
    if min_age is None:
        min_age = -99999.0

    return data_to_process[
        (data_to_process["age_range_min"].astype(int) < max_age)
        & (data_to_process["age_range_max"].astype(int) >= min_age)
    ]


def sex_filter(data_to_process: DataFrame, sex_list: list) -> DataFrame:
    """Sex filter

    Args:
        data_to_process (DataFrame): _description_
        sex_list (list): _description_

    Returns:
        DataFrame: _description_
    """
    sex_values = []
    for proc_sex in sex_list:
        sex_values.append(SEX_INDEX[proc_sex])
    return data_to_process[data_to_process["sex"].isin(sex_values)]


def ethnicity_filter(data_to_process: DataFrame, ethnicity_list: list) -> DataFrame:
    """Ethnicity filter

    Args:
        data_to_process (DataFrame): _description_
        ethnicity_list (list): _description_

    Returns:
        DataFrame: _description_
    """
    ethnicity_values = []
    for proc_ethnicity in ethnicity_list:
        ethnicity_values.append(ETHNICITY_INDEX[proc_ethnicity])
    return data_to_process[data_to_process["ethnicity"].isin(ethnicity_values)]


def vaccine_filter(data_to_process: DataFrame, vaccine_list: list) -> DataFrame:
    """Vaccine filter

    Args:
        data_to_process (DataFrame): _description_
        vaccine_list (list): _description_

    Returns:
        DataFrame: _description_
    """
    return data_to_process[data_to_process["vaccine"].isin(vaccine_list)]


def contacts_filter(
    agents_data: DataFrame,
    interaction_data: DataFrame,
    filtered_data: DataFrame,
    contacts_cfg: dict,
) -> DataFrame:
    """Filter data based on the required contacts

    Args:
        agents_data (DataFrame): _description_
        interaction_data (DataFrame): _description_
        filtered_data (DataFrame): _description_
        contacts_cfg (dict): _description_

    Returns:
        DataFrame: _description_
    """
    all_index = agents_data.index[agents_data["id"].isin(filtered_data["id"])]
    col1_counts = (
        interaction_data[interaction_data["id_x"].isin(all_index)]
        .groupby("spec_x")["id_x"]
        .value_counts()
    )

    df = col1_counts.to_frame()
    df = df.rename(columns={"id_x": "occurance"})
    df["group_and_idx"] = df.index
    df = df.reset_index(drop=True)

    df[["group", "idx"]] = DataFrame(df.group_and_idx.tolist(), index=df.index)
    df = df.drop("group_and_idx", axis=1)
    df["id"] = filtered_data.loc[df["idx"]]["id"].values
    df = df.drop("idx", axis=1)

    loc_index_swapped = dict(zip(LOC_INDEX.values(), LOC_INDEX.keys()))
    for index, row in df.iterrows():
        df.loc[index, "group_replaced"] = loc_index_swapped[row["group"]]

    df = df.drop("group", axis=1)
    df["is_within_range"] = False
    for index, row in df.iterrows():
        group = row["group_replaced"]
        if_enable = contacts_cfg[group]["enable"]

        if not if_enable:
            continue

        min_range = contacts_cfg[group]["min"]
        max_range = contacts_cfg[group]["max"]
        if min_range is None:
            min_range = -999999.0
        if max_range is None:
            max_range = 999999.0
        df.loc[index, "is_within_range"] = (row["occurance"] >= min_range) and (
            row["occurance"] <= max_range
        )

    # Select the rows from the DataFrame X where the value of the new column is True
    df = df.loc[df["is_within_range"]]

    df = df.drop("is_within_range", axis=1)
    df = df.rename(columns={"group_replaced": "group"})

    return df


def agents_filter(cfg: dict, data: dict) -> DataFrame:
    """Agents filter

    Args:
        cfg (dict): _description_
        data (dict): _description_

    Returns:
        DataFrame: _description_
    """
    dfs = []
    for proc_area_info in cfg["all_agents"]:
        for proc_agents_info in cfg["all_agents"][proc_area_info]:
            proc_agents_data = data["agents_data"][data["agents_data"]["area"] == proc_area_info]
            proc_agents_info_input = proc_agents_info["agents"]

            # Obtain age
            proc_agents_data = age_filter(
                proc_agents_data,
                proc_agents_info_input["age"]["max"],
                proc_agents_info_input["age"]["min"],
            )

            # Obtain sex
            proc_agents_data = sex_filter(proc_agents_data, proc_agents_info_input["sex"])

            # Obtain ethnicity
            proc_agents_data = ethnicity_filter(
                proc_agents_data, proc_agents_info_input["ethnicity"]
            )

            # Obtain vaccine:
            proc_agents_data = vaccine_filter(proc_agents_data, proc_agents_info_input["vaccine"])

            # Obtain contacts
            all_filtered_output = contacts_filter(
                data["agents_data"],
                data["interaction_data"],
                proc_agents_data,
                proc_agents_info_input["contacts"],
            )

            random_sample = all_filtered_output.sample(n=proc_agents_info_input["num"])
            dfs.append(random_sample)

    return concat(dfs)


def write_out_df(workdir: str, data: DataFrame):
    data.to_csv(join(workdir, "agents_filter.csv"))
