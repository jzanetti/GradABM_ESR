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

from process.input.create_inputs import (
    get_agents,
    get_diary_data,
    get_interactions,
    get_sa2_from_dhb,
    read_june_nz_inputs,
    write_target,
)
from process.utils.utils import read_cfg, setup_logging


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
        "--workdir",
        required=True,
        help="Working directory, e.g., where the output will be stored",
    )

    parser.add_argument(
        "--diary_path",
        required=True,
        help="Diary data path",
    )

    parser.add_argument(
        "--synpop_path",
        required=True,
        help="Synthetic population data path",
    )

    parser.add_argument(
        "--sa2_dhb_map_path",
        required=False,
        default=None,
        help="Data path for SA2 to DHB map",
    )

    parser.add_argument(
        "--target_path",
        required=False,
        default=None,
        help="Target data path",
    )

    parser.add_argument(
        "--cfg_path", required=True, help="Configuration path for the input ..."
    )

    parser.add_argument(
        "--dhb_list",
        nargs="+",
        help="DHB list to be used",
        required=False,
        default=None,
    )

    """
    return parser.parse_args(
        [
            "--june_nz_data",
            # "/tmp/june_realworld_auckland/interaction_output.parquet",
            "/tmp/june_realworld_auckland/interaction_output",
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
    """
    return parser.parse_args()


def main(
    workdir: str,
    diary_path: str,
    synpop_path: str,
    sa2_dhb_map_path: str,
    target_path: str,
    cfg_path: str,
    dhb_list: list,
):
    """Run June model for New Zealand

    Args:
        workdir (str): Working directory
        diary_path (str): _description_
        synpop_path (str): _description_
        sa2_dhb_map_path (str): _description_
        target_path (str): _description_
        cfg_path (str): _description_
        dhb_list (list): _description_
    """

    if not exists(workdir):
        makedirs(workdir)

    logger = setup_logging(workdir)

    logger.info("Reading configuration ...")
    cfg = read_cfg(cfg_path)

    logger.info("Creating target ...")
    write_target(workdir, target_path, dhb_list)

    logger.info("Getting SA2 from DHB")
    sa2 = get_sa2_from_dhb(sa2_dhb_map_path, dhb_list)

    logger.info("Getting agents and diget_agent_and_diary_dataaries ...")
    diary = get_diary_data(synpop_path, diary_path)

    logger.info("Creating interactions ...")
    get_interactions(
        diary,
        sa2,
        workdir,
        cfg["interaction_ratio"],
        max_interaction_for_each_venue=None,
    )

    logger.info("Job done ...")


if __name__ == "__main__":
    args = setup_parser()
    main(
        args.workdir,
        args.diary_path,
        args.synpop_path,
        args.sa2_dhb_map_path,
        args.target_path,
        args.cfg_path,
        args.dhb_list,
    )
