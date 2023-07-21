"""
Usage: cli_create_input --workdir /tmp/june_nz --cfg june.cfg
Author: Sijin Zhang
Contact: sijin.zhang@esr.cri.nz

Description: 
    This is a wrapper to convert JUNE-NZ interaction data to GradABM
"""

import argparse
from os import makedirs
from os.path import exists

from input.create_inputs import get_agents, get_interactions, read_june_nz_inputs
from utils.utils import read_cfg, setup_logging


def get_example_usage():
    example_text = """example:
        * create_input --workdir /tmp/gradabm_input
                       --cfg gradabm_exp.cfg
        """
    return example_text


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Creating inputs of GradABM from JUNE-NZ",
        epilog=get_example_usage(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--june_nz_data",
        required=True,
        help="June-NZ data input",
    )
    parser.add_argument(
        "--workdir", required=True, help="Working directory, e.g., where the output will be stored"
    )
    parser.add_argument("--cfg", required=True, help="Configuration path for the input ...")

    return parser.parse_args(
        [
            "--june_nz_data",
            "interaction_output.parquet",
            "--workdir",
            "/tmp/gradabm_esr_input2",
            "--cfg",
            "cfg/sample_cfg/input_exp.yml",
        ]
    )


def main():
    """Run June model for New Zealand"""
    args = setup_parser()

    if not exists(args.workdir):
        makedirs(args.workdir)

    logger = setup_logging(args.workdir)

    logger.info("Reading configuration ...")
    cfg = read_cfg(args.cfg)

    logger.info("Getting JUNE-NZ data as input ...")
    data = read_june_nz_inputs(args.june_nz_data)

    logger.info("Getting agents ...")
    get_agents(data, args.workdir, cfg["vaccine_ratio"])

    logger.info("Creating interactions ...")
    get_interactions(data, args.workdir)

    logger.info("Job done ...")


if __name__ == "__main__":
    main()
