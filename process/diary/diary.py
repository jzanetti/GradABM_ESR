from datetime import datetime, timedelta
from logging import getLogger
from os.path import join
from pickle import dump as pickle_dump

import ray
from numpy import array as numpy_array
from numpy.random import choice as numpy_choice
from numpy.random import normal as numpy_normal
from pandas import DataFrame
from pandas import read_csv as pandas_read_csv

from process.diary import DIARY_CFG

logger = getLogger()


def read_synthetic_pop(data_path: str) -> DataFrame:
    """Read synthetic population

    Args:
        data_path (str): population path

    Returns:
        DataFrame: sys population
    """
    return pandas_read_csv(data_path)


def create_diary_single_person(
    ref_time: datetime = datetime(1970, 1, 1, 0),
    time_var: numpy_array = numpy_normal(0.0, 2.0, 100),
    activities: dict = DIARY_CFG["default"],
):
    ref_time_start = ref_time
    ref_time_end = ref_time + timedelta(hours=24)

    output = {}
    ref_time_proc = ref_time_start
    while ref_time_proc <= ref_time_end:
        # Get all activities that can be chosen at this time
        available_activities = []
        for activity in activities:
            for start, end in activities[activity]["time_ranges"]:
                start2 = ref_time_start + timedelta(
                    hours=(start - abs(numpy_choice(time_var)))
                )
                end2 = ref_time_start + timedelta(
                    hours=(end + abs(numpy_choice(time_var)))
                )

                if start2 <= ref_time_proc < end2:
                    available_activities.append(activity)

        if available_activities:
            # Choose an activity based on the probabilities
            available_probabilities = [
                activities[proc_activity]["prob"]
                for proc_activity in available_activities
            ]

            # scale up the probability to 1.0
            available_probabilities = numpy_array(available_probabilities)
            available_probabilities /= available_probabilities.sum()

            activity = numpy_choice(available_activities, p=available_probabilities)

            # Add the activity to the diary
            output[ref_time_proc.hour] = activity

        else:
            output[ref_time_proc.hour] = numpy_choice(list(activities.keys()))

        ref_time_proc += timedelta(hours=1)

    return output


@ray.remote
def create_diary_remote(
    syspop_data: DataFrame, ncpu: int, print_log: bool
) -> DataFrame:
    """Create diaries in parallel processing

    Args:
        workdir (str): Working directory
        syspop_data (DataFrame): Synthetic population
        ncpu (int): Number of CPUs in total
            (this is just for displaying the progress)
    """
    return create_diary(syspop_data, ncpu, print_log)


def create_diary(syspop_data: DataFrame, ncpu: int, print_log: bool) -> DataFrame:
    """Create diaries

    Args:
        workdir (str): Working directory
        syspop_data (DataFrame): Synthetic population
        ncpu (int): Number of CPUs in total
            (this is just for displaying the progress)
    """

    all_diaries = {"id": []}

    all_diaries = {proc_hour: [] for proc_hour in range(24)}
    total_people = len(syspop_data)
    for i in range(total_people):
        proc_people = syspop_data.iloc[i]
        if print_log:
            print(
                f"Processing [{i}/{total_people}]x{ncpu}: {100.0 * round(i/total_people, 2)}x{ncpu} %"
            )

        output = create_diary_single_person(
            activities=DIARY_CFG.get(
                "worker"
                if isinstance(proc_people["company"], str)
                else "student"
                if isinstance(proc_people["school"], str)
                else "default"
            )
        )

        all_diaries["id"].append(i)

        for j in output:
            all_diaries[j].append(output[j])

    all_diaries = DataFrame.from_dict(all_diaries)

    return all_diaries

    # pickle_dump({"diaries": all_diaries}, open(join(workdir, "diaries.pickle"), "wb"))
