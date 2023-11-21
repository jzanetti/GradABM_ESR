"""
Usage: cli_agents --workdir /tmp/june_nz --cfg agents.cfg
Author: Sijin Zhang
Contact: sijin.zhang@esr.cri.nz

Description: 
    This is a wrapper to identify agents meet certain requirement
"""

import argparse
from os import makedirs
from os.path import exists

from process.input.agents_filter import agents_filter, obtain_agents_info, write_out_df
from process.utils.utils import read_cfg, setup_logging


def get_example_usage():
    example_text = """example:
        * cli_agents --workdir /tmp/gradabm_input
                       --cfg agents.cfg
        """
    return example_text


def setup_parser():
    parser = argparse.ArgumentParser(
        description="This is a wrapper to identify agents meet certain requirement",
        epilog=get_example_usage(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--workdir",
        required=True,
        help="Working directory, e.g., where the output will be stored",
    )
    parser.add_argument("--agents_data", required=True, help="Agents data in parquet")
    parser.add_argument(
        "--interaction_data", required=True, help="Interaction data in parquet"
    )
    parser.add_argument(
        "--cfg", required=True, help="Configuration path for identifying agents ..."
    )

    return parser.parse_args(
        [
            "--workdir",
            "exp/policy_paper/initial_agents",
            "--agents_data",
            "exp/policy_paper/input/agents.parquet",
            "--interaction_data",
            "exp/policy_paper/input/interaction_graph_cfg_member_0.parquet",
            "--cfg",
            "data/measles/base/initial_agents.yml",
        ]
    )

    return parser.parse_args()


def main(workdir, agents_data_path, interaction_data_path, cfg):
    """Create agents filter"""

    if not exists(workdir):
        makedirs(workdir)

    logger = setup_logging(workdir)

    logger.info("Reading configuration ...")
    cfg = read_cfg(cfg)

    logger.info("Obtaining agents filter input ...")
    data = obtain_agents_info(agents_data_path, interaction_data_path)

    logger.info("Agents filter ...")
    data = agents_filter(cfg, data)

    logger.info("Export agents ...")
    write_out_df(workdir, data)

    logger.info("done")


if __name__ == "__main__":
    args = setup_parser()
    main(
        args.workdir,
        args.agents_data,
        args.interaction_data,
        args.cfg,
    )
