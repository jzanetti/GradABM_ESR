from pickle import dump as pickle_dump
from pickle import load as pickle_load
from random import randint as random_randint

from funcs import PEOPLE_MOVEMENT
from funcs.func_diary import get_diary_data
from funcs.func_util import perturbate_latlon
from pandas import read_csv as pandas_read_csv
from pandas import read_parquet as pandas_read_parquet


def read_syspop(syspop_path: str, syspop_address_path: str, sample_size=None):
    base_data = pandas_read_csv(syspop_path)
    address_data = pandas_read_csv(syspop_address_path)
    if sample_size is not None:
        base_data = base_data.sample(sample_size)
    return {
        "base": base_data,
        "address": address_data,
    }


def read_diaries(diary_path: str):
    return pandas_read_parquet(diary_path)


def extract_data(extracted_data_path: str):
    syspop_data = read_syspop(
        syspop_path="etc/dashboard/testdata/syspop_base.csv",
        syspop_address_path="etc/dashboard/testdata/syspop_location.csv",
        sample_size=None,
    )
    diary_data = read_diaries(diary_path="etc/dashboard/testdata/diaries.parquet")
    diary_location = get_diary_data(
        syspop_data["base"], syspop_data["address"], diary_data
    )

    diary_location = perturbate_latlon(diary_location)

    syspop_address_data = syspop_data["address"]

    pickle_dump(
        {"diary_location": diary_location, "syspop_address_data": syspop_address_data},
        open(extracted_data_path, "wb"),
    )


def preproc_data(extracted_data_path: str, output_path: str):
    preproc_data = pickle_load(open(extracted_data_path, "rb"))
    diary_location = preproc_data["diary_location"]
    syspop_address_data = preproc_data["syspop_address_data"]

    total_data_to_process = (
        len(PEOPLE_MOVEMENT["place"]) * 24 * len(PEOPLE_MOVEMENT["sample_size"])
    )
    index = 0
    output = {}
    for proc_sample_size in PEOPLE_MOVEMENT["sample_size"]:
        if proc_sample_size["value"] not in output:
            output[proc_sample_size["value"]] = {}

        for proc_place in PEOPLE_MOVEMENT["place"]:
            if proc_place["value"] not in output[proc_sample_size["value"]]:
                output[proc_sample_size["value"]][proc_place["value"]] = {}

            proc_data = diary_location[
                diary_location["type"] == proc_place["value"]
            ].sample(proc_sample_size["value"])

            for proc_hr in range(24):
                output[proc_sample_size["value"]][proc_place["value"]][
                    proc_hr
                ] = proc_data[proc_data["hour"] == str(proc_hr)]

                print(f" - preproc_data: completed: {index} / {total_data_to_process}")
                index += 1

    pickle_dump(
        {"syspop_address_data": syspop_address_data, "sampled_data": output},
        open(output_path, "wb"),
    )
