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
from os.path import exists

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
        "--param_path",
        required=True,
        help="Parameters path",
    )
    parser.add_argument(
        "--output_info_path",
        required=True,
        help="Output information path",
    )
    parser.add_argument("--param_model_path", required=True, help="Trained model path")

    parser.add_argument(
        "--cfg",
        required=True,
        help="Configuration path for the model (e.g., scaling factor), "
        "e.g., gradadm_exp.cfg (This is used to create a customized "
        "ABM but using trained parameters)",
    )

    parser.add_argument(
        "--agents_data",
        required=True,
        help="Agents data in parquet (This is used to create a customized "
        "ABM but using trained parameters)",
    )
    parser.add_argument(
        "--interaction_data",
        required=True,
        help="Interaction data in parquet (This is used to create a customized "
        "ABM but using trained parameters)",
    )
    parser.add_argument(
        "--target_data",
        required=True,
        help="Target data in CSV (This is used to create a customized "
        "ABM but using trained parameters)",
    )

    return parser.parse_args(
        [
            "--workdir",
            "/tmp/gradabm_esr_pred2",
            "--param_path",
            "/tmp/gradabm_esr/test1/params.p",
            "--output_info_path",
            "/tmp/gradabm_esr/test1/output_info.p",
            "--param_model_path",
            "/tmp/gradabm_esr/test1/param_model.model",
            "--cfg",
            "cfg/sample_cfg/gradam_exp.yml",
            "--agents_data",
            "/tmp/gradabm_esr_input2/agents.parquet",
            "--interaction_data",
            "/tmp/gradabm_esr_input2/interaction_graph_cfg.parquet",
            "--target_data",
            "data/exp4/targets3.csv",
        ]
    )


def main():
    """Run June model for New Zealand"""
    args = setup_parser()

    if not exists(args.workdir):
        makedirs(args.workdir)

    logger = setup_logging(args.workdir)

    logger.info("Loading trained model ...")
    trained_output = load_outputs(args.param_path, args.output_info_path, args.param_model_path)

    logger.info("Reading configuration ...")
    cfg = read_cfg(args.cfg)

    logger.info("Getting model input ...")
    model_inputs = prep_model_inputs(
        args.agents_data, args.interaction_data, args.target_data, cfg["interaction"]
    )

    logger.info("Building ABM ...")
    abm = build_abm(model_inputs["all_agents"], model_inputs["all_interactions"], cfg["infection"])

    logger.info("Creating prediction ...")
    predictions = forward_abm(
        trained_output["param"]["param_with_smallest_loss"],
        trained_output["param_model"].param_info(),
        abm,
        trained_output["output_info"]["total_timesteps"],
        save_records=True,
    )

    logger.info("Output processing ...")
    output = postproc(
        trained_output["param_model"], predictions, trained_output["output_info"]["target"]
    )

    logger.info("Visualization ...")
    plot_diags(
        args.workdir,
        output,
        trained_output["output_info"]["epoch_loss_list"],
        apply_norm=False,
    )

    logger.info("Job done ...")


if __name__ == "__main__":
    main()
