from os import makedirs
from os.path import exists, join
from typing import Dict, List, Optional

from process.input.agents_and_interaction import (
    create_agents_and_interactions,
    get_diary_data,
    get_sa2_from_dhb,
    write_target,
)
from process.utils.utils import read_cfg, setup_logging


def input_wrapper(
    workdir: str,
    synpop_path: str,
    diary_path: str,
    cfg_path: str,
    target_path: Optional[str] = None,
    target_index_range: Optional[Dict] = None,
    geography_ancillary_path: Optional[str] = None,
    dhb_list: Optional[List] = None,
):
    """Input processing for New Zealand

    Args:
        workdir (str): Working directory
        diary_path (str): Diary data path (e.g., the one created by Syspop)
        synpop_path (str): Synthetic population data path (e.g., the one created by Syspop)
        sa2_dhb_map_path (str): The data maping SA2 and DHB for New Zealand
        target_path (str): Target data to be trained towards
        cfg_path (str): Input data path
        dhb_list (list): DHB to be applied
        target_index_range (dict ot None): the index range for the target, e.g., {"start": 13, "end": -1}
    """

    if not exists(workdir):
        makedirs(workdir)

    logger = setup_logging(workdir)

    logger.info("Reading configuration ...")
    cfg = read_cfg(cfg_path)

    logger.info("Creating target ...")
    write_target(workdir, target_path, dhb_list, target_index_range)

    logger.info("Getting SA2 from DHB")
    sa2 = get_sa2_from_dhb(geography_ancillary_path, dhb_list)

    total_venue_ens = cfg["ensemble"]["venue"]
    total_interaction_ens = cfg["ensemble"]["interaction"]

    for venue_id in range(total_venue_ens):

        logger.info(f"Getting agents and diary: {venue_id} / {total_venue_ens - 1} ...")

        proc_agents_and_diary = get_diary_data(synpop_path, diary_path)

        for interaction_id in range(cfg["ensemble"]["interaction"]):

            logger.info(
                f"Creating interactions {venue_id} ({total_venue_ens - 1}) / "
                f"{interaction_id} ({total_interaction_ens  - 1})"
            )
            agents_and_interactions_data = create_agents_and_interactions(
                proc_agents_and_diary,
                sa2,
                cfg["interaction_ratio"],
                cfg["runtime_attributes"],
                max_interaction_for_each_venue=None,
            )
            agents_and_interactions_data["interactions"].to_parquet(
                join(
                    workdir,
                    f"interaction_graph_cfg_member_{venue_id}_{interaction_id}.parquet",
                )
            )

        if venue_id == 0:
            agents_and_interactions_data["agents"].to_parquet(
                join(workdir, f"agents.parquet")
            )

    logger.info("Job done ...")
