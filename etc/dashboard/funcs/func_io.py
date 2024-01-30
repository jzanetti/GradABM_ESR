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
