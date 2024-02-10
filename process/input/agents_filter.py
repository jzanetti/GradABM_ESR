from json import load as json_load
from pickle import load as pickle_load

from numpy import NaN as numpy_NaN
from pandas import DataFrame
from pandas import concat as pandas_concat
from pandas import read_csv as pandas_read_csv


def prepare_agents(syspop_path: str, diary_path: str, agent_paths: list) -> dict:
    """_summary_

    Args:
        syspop_path (str): _description_
        diary_path (str): _description_
        agent_paths (list): _description_

    Returns:
        dict: _description_
    """
    all_agents = []
    for proc_agent_path in agent_paths:
        with open(proc_agent_path) as json_file:
            all_agents.append(json_load(json_file))

    syspop_data = pandas_read_csv(syspop_path)
    diary_data = pickle_load(open(diary_path, "rb"))

    return {
        "syspop_data": syspop_data,
        "diary_data": diary_data,
        "all_agents": all_agents,
    }


def agents_filter(input_data: dict) -> DataFrame:
    """Data filter

    Args:
        syspop_data_keys (list): _description_
        proc_agent (dict): _description_
        syspop_data (DataFrame): _description_

    Returns:
        str or numpy_NaN: _description_
    """

    def _data_filter(
        proc_key: str, proc_agent: dict, syspop_data: DataFrame
    ) -> str or numpy_NaN:
        """Data filter

        Args:
            syspop_data_keys (list): _description_
            proc_agent (dict): _description_
            syspop_data (DataFrame): _description_

        Returns:
            str or numpy_NaN: _description_
        """
        if proc_key == "area":
            return proc_agent["locations"]["household"]["sa2"]
        elif proc_key == "area_work":
            if proc_agent["locations"]["work"] is not None:
                return proc_agent["locations"]["work"]["sa2"]
            else:
                return numpy_NaN
        elif proc_key == "travel_mode_work":
            if proc_agent["locations"]["work"] is not None:
                return proc_agent["locations"]["work"]["travel_mode"]
            else:
                return numpy_NaN
        elif proc_key == "public_transport_trip":
            if proc_agent["locations"]["work"] is not None:
                return proc_agent["locations"]["work"]["public_transport_trip"]
            else:
                return numpy_NaN
        elif proc_key in ["primary_hospital", "secondary_hospital"]:
            return numpy_NaN
        elif proc_key == "company":
            if proc_agent["locations"]["work"] is not None:
                company_name = (
                    syspop_data[
                        syspop_data["company"].str.startswith(
                            f"{proc_agent['locations']['work']['work_code']}_",
                            na=False,
                        )
                        & syspop_data["company"].str.endswith(
                            f"{proc_agent['locations']['work']['sa2']}", na=False
                        )
                    ]
                    .sample(1)["company"]
                    .values[0]
                )
                return company_name
            else:
                return numpy_NaN

        elif proc_key == "school":
            if proc_agent["locations"]["school"] is not None:
                school_name = (
                    syspop_data[
                        (
                            syspop_data["area"]
                            == proc_agent["locations"]["household"]["sa2"]
                        )
                        & (syspop_data["age"] == proc_agent["basic"]["age"])
                    ]
                    .sample(1)["school"]
                    .values[0]
                )
                return school_name
            else:
                return numpy_NaN

        elif proc_key in ["age", "gender", "ethnicity", "social_economics"]:
            return proc_agent["basic"][proc_key]
        elif proc_key == "household":
            household_name = syspop_data[
                syspop_data["household"].str.startswith(
                    f"{proc_agent['locations']['household']['sa2']}_{proc_agent['locations']['household']['adults']}_{proc_agent['locations']['household']['children']}"
                )
            ].sample(1)["household"]
            return household_name
        elif proc_key in ["supermarket", "restaurant"]:
            return (
                syspop_data[
                    (syspop_data["area"] == proc_agent["locations"][proc_key]["sa2"])
                ]
                .sample(1)[proc_key]
                .values[0]
            )

    syspop_data_keys = list(input_data["syspop_data"].columns)
    agents_data = {}
    for proc_key in syspop_data_keys:
        agents_data[proc_key] = []

    for index, proc_agent in enumerate(input_data["all_agents"]):
        agents_data["id"].append(input_data["syspop_data"]["id"].max() + index + 1)

        for proc_key in syspop_data_keys:
            if proc_key == "id":
                continue

            agents_data[proc_key].append(
                _data_filter(proc_key, proc_agent, input_data["syspop_data"])
            )

    agents_data = DataFrame.from_dict(agents_data)
    agents_data["index"] = agents_data["id"]
    agents_data = agents_data.set_index("index", inplace=False)
    agents_data.index.name = None
    updated_syspop_data = pandas_concat([input_data["syspop_data"], agents_data])
    return updated_syspop_data


def diary_filter(input_data: dict) -> dict:

    diary_data = input_data["diary_data"]["diaries"]

    diary_output = {}
    for proc_hr in range(24):
        if "id" not in diary_output:
            diary_output["id"] = []

        if proc_hr not in diary_output:
            diary_output[proc_hr] = []
        for index, proc_agent in enumerate(input_data["all_agents"]):

            if proc_hr == 0:
                diary_output["id"].append(diary_data["id"].max() + index + 1)

            try:
                diary_output[proc_hr].append(proc_agent["diary"][str(proc_hr)])
            except KeyError:
                diary_output[proc_hr].append(proc_agent["diary"]["default"])

    diary_output = DataFrame.from_dict(diary_output)

    updated_diary_data = diary_data.append(diary_output, ignore_index=True)

    return updated_diary_data
