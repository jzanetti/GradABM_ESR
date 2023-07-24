"""
Usage: cli_predict --workdir  /tmp/gradabm_esr_pred
                   --param_path /tmp/gradabm_esr/test1/params.p
                   --output_info_path /tmp/gradabm_esr/test1/output_info.p
                   --param_model_path /tmp/gradabm_esr/test1/param_model.model
                   --abm_model_path /tmp/gradabm_esr/test1/abm_model.model
Author: Sijin Zhang
Contact: sijin.zhang@esr.cri.nz

Description: 
    This is a wrapper to run the prediction
"""

import argparse
from os import makedirs
from os.path import exists, join

from input import INTERACTION_ENS_MEMBERS
from model.abm import build_abm, forward_abm
from model.diags import load_outputs, plot_diags
from model.postp import postproc
from model.prep import prep_model_inputs
from utils.utils import read_cfg, setup_logging


def get_example_usage():
    example_text = """example:
        * cli_predict # ---------------------------
                      # Trained parameters
                      # ---------------------------
                      --workdir  /tmp/gradabm_esr_pred
                      --param_path /tmp/gradabm_esr/test1/params.p
                      --output_info_path /tmp/gradabm_esr/test1/output_info.p
                      --param_model_path /tmp/gradabm_esr/test1/param_model.model
                      # ---------------------------
                      # The following are used to construct a new ABM model to 
                      # do prediction using previously trained parameters
                      # ---------------------------
                      --cfg cfg/sample_cfg/gradam_exp.yml
                      --agents_data /tmp/gradabm_esr_input/agents.parquet
                      --interaction_data /tmp/gradabm_esr_input/interaction_graph_cfg.parquet
                      --target_data data/exp4/targets3.csv

        """
    return example_text


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Predicting GradABM model",
        epilog=get_example_usage(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--workdir", required=True, help="Working directory, e.g., where the output will be stored"
    )

    parser.add_argument(
        "--model_base_dir",
        nargs="+",
        help="Model base dir path (e.g., /tmp/manukau_measles_2019)",
        required=True,
    )

    parser.add_argument(
        "--cfg",
        required=True,
        help="Configuration path for the model (e.g., scaling factor), "
        "e.g., gradadm_exp.cfg (This is used to create a customized "
        "ABM but using trained parameters)",
    )

    """
        [
            "--workdir",
            "/tmp/gradabm_esr_pred2",
            "--model_exp",
            "/tmp/gradabm_esr/test1"
            "--cfg",
            "cfg/sample_cfg/gradam_exp.yml",
            "--agents_data",
            "/tmp/gradabm_esr_input2/agents.parquet",
            "--interaction_data",
            "/tmp/gradabm_esr_input2/interaction_graph_cfg.parquet",
            "--target_data",
            "data/exp4/targets3.csv",
        ]
    """

    return parser.parse_args(
        [
            "--workdir",
            "/tmp/gradabm_esr_pred_auckland",
            "--model_base_dir",
            "/tmp/manukau_measles_2019",
            "--cfg",
            "data/measles/auckland/gradam_exp.yml",
        ]
    )


def main(workdir, cfg, model_base_dir, replace_agents_with: str or None = None):
    if not exists(workdir):
        makedirs(workdir)

    logger = setup_logging(workdir)

    logger.info("Reading configuration ...")
    cfg = read_cfg(cfg)

    logger.info("Getting data path ...")
    model_exps = join(model_base_dir, "model", "member_{proc_member}")

    if replace_agents_with:
        agents_data = replace_agents_with
    else:
        agents_data = join(model_base_dir, "input", "agents.parquet")

    interaction_data = join(
        model_base_dir, "input", "interaction_graph_cfg_member_{proc_member}.parquet"
    )
    target_data = join(model_base_dir, "input", "output.csv")

    logger.info("Obtaininig data ...")
    outputs = []
    epoch_losses = []
    for proc_member in range(INTERACTION_ENS_MEMBERS):  # INTERACTION_ENS_MEMBERS:
        proc_model_member_dir = model_exps.format(proc_member=proc_member)
        param_path = join(proc_model_member_dir, "params.p")
        output_info_path = join(proc_model_member_dir, "output_info.p")
        param_model_path = join(proc_model_member_dir, "param_model.model")

        logger.info("Loading trained model ...")
        trained_output = load_outputs(param_path, output_info_path, param_model_path)

        logger.info("Getting model input ...")
        model_inputs = prep_model_inputs(
            agents_data,
            interaction_data.format(proc_member=proc_member),
            target_data,
            cfg["interaction"],
        )

        logger.info("Building ABM ...")
        abm = build_abm(
            model_inputs["all_agents"], model_inputs["all_interactions"], cfg["infection"]
        )

        logger.info("Creating prediction ...")
        predictions = forward_abm(
            trained_output["param"]["param_with_smallest_loss"],
            trained_output["param_model"].param_info(),
            abm,
            trained_output["output_info"]["total_timesteps"],
            save_records=True,
        )

        logger.info("Output processing ...")
        outputs.append(
            postproc(
                trained_output["param_model"], predictions, trained_output["output_info"]["target"]
            )
        )

        epoch_losses.append(trained_output["output_info"]["epoch_loss_list"])

    logger.info("Visualization ...")
    plot_diags(
        workdir,
        outputs,
        epoch_losses,
        cfg["temporal_res"],
        apply_norm=False,
    )

    logger.info("Job done ...")


if __name__ == "__main__":
    args = setup_parser()
    main(args.workdir, args.cfg, args.model_base_dir)
