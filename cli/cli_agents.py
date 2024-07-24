"""
Usage: cli_agents --workdir /tmp/june_nz --cfg agents.cfg
Author: Sijin Zhang
Contact: sijin.zhang@esr.cri.nz

Description: 
    This is a wrapper to identify agents meet certain requirement
"""

import argparse

from process.input.agents_filter import agents_filter, diary_filter, prepare_agents


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

    parser.add_argument("--syspop_path", required=True, help="Agents data in parquet")
    parser.add_argument("--diary_path", required=True, help="Diary data path")
    parser.add_argument("--agent_paths", nargs="+", help="List of agents")

    return parser.parse_args(
        [
            "--syspop_path",
            "/tmp/syspop_test/Auckland/syspop_base.csv",
            "--diary_path",
            "/tmp/gradabm_esr/Auckland/diaries.pickle",
            "--agent_paths",
            "data/measles_v2/agents_from_tracking/agent_1.json",
            "data/measles_v2/agents_from_tracking/agent_2.json",
            "data/measles_v2/agents_from_tracking/agent_3.json",
        ]
    )

    return parser.parse_args()


def main(syspop_path: str, diary_path: str, agent_paths: list):
    """Create agents filter"""

    input_data = prepare_agents(syspop_path, diary_path, agent_paths)
    updated_syspop_data = agents_filter(input_data)
    updated_diary_data = diary_filter(input_data)

    updated_syspop_data.to_csv(syspop_path.replace(".csv", "_updated.csv"))
    updated_diary_data.to_csv(diary_path.replace(".pickle", "_updated.pickle"))


if __name__ == "__main__":
    args = setup_parser()
    main(args.syspop_path, args.diary_path, args.agent_paths)
