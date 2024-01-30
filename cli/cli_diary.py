"""
Usage: cli_create_input --workdir /tmp/june_nz --cfg june.cfg
Author: Sijin Zhang
Contact: sijin.zhang@esr.cri.nz

Description: 
    This is a wrapper to create diary using the data from Syspop
"""
import argparse
from datetime import datetime
from os import environ as os_env
from os import makedirs
from os.path import exists, join
from pickle import dump as pickle_dump

import ray
from pandas import concat as pandas_concat
from pandas import cut as pandas_cut

from process.diary.diary import create_diary, create_diary_remote, read_synthetic_pop
from process.utils.utils import setup_logging

os_env["RAY_DEDUP_LOGS"] = "0"


def get_example_usage():
    example_text = """example:
        * create_input --workdir /tmp/gradabm_input
                       --cfg gradabm_exp.cfg
        """
    return example_text


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Creating diary from synthetic population",
        epilog=get_example_usage(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--synthetic_pop_path",
        required=True,
        help="Synthentic population data path (e.g., from Syspop)",
    )

    parser.add_argument(
        "--workdir",
        required=True,
        help="Working directory, e.g., where the output will be stored",
    )

    parser.add_argument(
        "--ncpu",
        required=False,
        default=8,
        help="Number of CPUs to be used",
    )

    return parser.parse_args(
        [
            "--synthetic_pop_path",
            "/tmp/syspop_test/Auckland/syspop_base.csv",
            "--workdir",
            "/tmp/gradabm_esr/Auckland_v3",
            "--ncpu",
            "1",
        ]
    )

    return parser.parse_args()


def main(workdir: str, synthetic_pop_path: str, ncpu: int):
    """Create synthetic diary"""

    if not exists(workdir):
        makedirs(workdir)

    start_t = datetime.utcnow()
    logger = setup_logging(workdir)

    logger.info(f"Reading synthetic population")
    syspop_data = read_synthetic_pop(synthetic_pop_path)

    syspop_data_partitions = [
        df for _, df in syspop_data.groupby(pandas_cut(syspop_data.index, ncpu))
    ]

    logger.info(f"Initiating Ray [cpu: {ncpu}]...")
    if ncpu > 1:
        ray.init(num_cpus=ncpu, include_dashboard=False)

    logger.info("Start processing diary ...")
    outputs = []
    for i, proc_syspop_data in enumerate(syspop_data_partitions):
        if ncpu == 1:
            outputs.append(create_diary(proc_syspop_data, ncpu, print_log=True))
        else:
            outputs.append(
                create_diary_remote.remote(proc_syspop_data, ncpu, print_log=i == 0)
            )

    if ncpu > 1:
        outputs = ray.get(outputs)
        ray.shutdown()

    outputs = pandas_concat(outputs, axis=0, ignore_index=True)
    end_t = datetime.utcnow()

    processing_mins = round((end_t - start_t).total_seconds() / 60.0, 2)

    pickle_dump({"diaries": outputs}, open(join(workdir, "diaries.pickle"), "wb"))

    logger.info(f"Diary created within {processing_mins} minutes ...")

    logger.info("Job done ...")


if __name__ == "__main__":
    args = setup_parser()
    main(args.workdir, args.synthetic_pop_path, int(args.ncpu))
