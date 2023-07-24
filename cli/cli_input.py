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

from input.create_inputs import (
    get_agents,
    get_interactions,
    get_sa2_from_dhb,
    read_june_nz_inputs,
    write_target,
)
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

    parser.add_argument(
        "--sa2_dhb_data",
        required=False,
        default=None,
        help="SA2 to DHB map",
    )

    parser.add_argument(
        "--target_data",
        required=False,
        default=None,
        help="Target data path",
    )

    parser.add_argument(
        "--dhb_list", nargs="+", help="DHB list to be used", required=False, default=None
    )
    return parser.parse_args(
        [
            "--june_nz_data",
            "/tmp/june_realworld_auckland/interaction_output.parquet",
            "--sa2_dhb_data",
            "data/dhb_and_sa2.parquet",
            "--workdir",
            "data/measles/auckland/inputs",
            "--cfg",
            "data/measles/auckland/input_exp.yml",
            "--dhb_list",
            # "Capital and Coast",
            # "Hutt Valley",
            "Counties Manukau",
            "--target_data",
            "data/measles_cases_2019.parquet",
        ]
    )


def main(
    workdir,
    cfg,
    june_nz_data,
    target_data,
    dhb_list,
    sa2_dhb_data,
):
    """Run June model for New Zealand"""

    if not exists(workdir):
        makedirs(workdir)

    logger = setup_logging(workdir)

    logger.info("Reading configuration ...")
    cfg = read_cfg(cfg)

    logger.info("Creating target ...")
    write_target(workdir, target_data, dhb_list)

    logger.info("Getting JUNE-NZ data as input ...")
    data = read_june_nz_inputs(june_nz_data)

    logger.info("Getting SA2 from DHB")
    sa2 = get_sa2_from_dhb(sa2_dhb_data, dhb_list)

    logger.info("Getting agents ...")
    agents = get_agents(data, sa2, workdir, cfg["vaccine_ratio"])

    logger.info("Creating interactions ...")
    get_interactions(data, agents, sa2, workdir)

    logger.info("Job done ...")


if __name__ == "__main__":
    args = setup_parser()
    main(
        args.workdir,
        args.cfg,
        args.june_nz_data,
        args.target_data,
        args.dhb_list,
        args.sa2_dhb_data,
    )
