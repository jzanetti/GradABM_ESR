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
        "--june_nz_data",
        required=True,
        help="June-NZ data input",
    )

    parser.add_argument(
        "--workdir",
        required=True,
        help="Working directory, e.g., where the output will be stored",
    )
    parser.add_argument(
        "--cfg", required=True, help="Configuration path for the input ..."
    )

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
    agents = get_agents(data, sa2, workdir, cfg["vaccine_ratio"], plot_agents=False)

    logger.info("Getting diaries ...")
    """
    # debug: read new data:
    from pickle import load as pickle_load
    from random import choice as random_choice

    from pandas import merge as pandas_merge
    from pandas import read_csv as pandas_read_csv

    new_agents = pandas_read_csv("/tmp/syspop_test/Auckland/syspop_base.csv")
    diary_data = pickle_load(open("/tmp/gradabm_esr/Auckland/diaries.pickle", "rb"))[
        "diaries"
    ]

    diary_data = diary_data[[12, "id"]]

    df_melted = diary_data.melt(id_vars="id", var_name="hour", value_name="spec")
    merged_df = pandas_merge(df_melted, new_agents, on="id", how="left")

    merged_df.loc[merged_df["spec"] == "household", "group"] = merged_df["household"]
    merged_df.loc[merged_df["spec"] == "supermarket", "group"] = merged_df[
        "supermarket"
    ]
    merged_df.loc[merged_df["spec"] == "restaurant", "group"] = merged_df["restaurant"]
    merged_df.loc[merged_df["spec"] == "travel", "group"] = merged_df[
        "travel_mode_work"
    ]
    merged_df.loc[merged_df["spec"] == "school", "group"] = merged_df["school"]
    merged_df.loc[merged_df["spec"] == "company", "group"] = merged_df["company"]

    merged_df["group"] = merged_df["group"].apply(
        lambda x: random_choice(x.split(",")) if "," in str(x) else x
    )

    # data = merged_df[["id", "area", "group", "spec", "hour"]]
    # agents = new_agents
    """
    logger.info("Creating interactions ...")
    get_interactions(
        data,
        agents,
        sa2,
        workdir,
        cfg["interaction_ratio"],
        max_interaction_for_each_venue=None,
    )["diaries"]

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
