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

from torch.cuda import empty_cache

from process.model.abm import build_abm
from process.model.diags import load_outputs, plot_diags
from process.model.postp import postproc_pred, write_output
from process.model.prep import prep_model_inputs
from process.model.wrapper import run_gradabm_wrapper
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
        "--workdir",
        required=True,
        help="Working directory, e.g., where the output will be stored",
    )

    parser.add_argument(
        "--model_base_dir",
        type=str,
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

    parser.add_argument(
        "--exp_id", required=True, help="Experiment name/id, e.g., base_exp"
    )

    parser.add_argument(
        "--model_id",
        required=False,
        default=0,
        help="Trained model ID [default: 0], "
        "for example, by default the paramters will be "
        "taken from member_0",
    )

    parser.add_argument(
        "--ens_id",
        required=False,
        default=0,
        help="Ensemble run ID [default: 0], "
        "the same model can produce differnet ensembles, this ID is used to identify different ens runs",
    )

    """
    return parser.parse_args(
        [
            "--workdir",
            "/tmp/gradabm_esr_pred_auckland",
            "--model_base_dir",
            "/tmp/manukau_measles_2019",
            "--cfg",
            "data/measles/auckland/gradam_exp.yml",
            "--exp_id",
            "base_exp",
            "--model_id",
            "0",
            "--ens_id",
            "0",
        ]
    )
    """
    return parser.parse_args()


def main(workdir, cfg, model_base_dir, proc_exp, model_id, ens_id):
    if not exists(workdir):
        makedirs(workdir)

    logger = setup_logging(workdir)

    logger.info("Reading configuration ...")
    cfg = read_cfg(cfg)

    logger.info("Getting data path ...")
    model_exps = join(model_base_dir, "model", "member_{proc_member}")

    logger.info("Getting agents ...")
    agents_data = join(model_base_dir, "input", "agents.parquet")

    interaction_data = join(
        model_base_dir, "input", "interaction_graph_cfg_member_{proc_member}.parquet"
    )
    target_data = join(model_base_dir, "input", "output.csv")

    logger.info(f"Obtaininig prediction for {proc_exp}...")
    proc_model_member_dir = model_exps.format(proc_member=model_id)
    param_path = join(proc_model_member_dir, "params.p")
    output_info_path = join(proc_model_member_dir, "output_info.p")
    param_model_path = join(proc_model_member_dir, "param_model.model")

    logger.info("Loading trained model ...")
    trained_output = load_outputs(param_path, output_info_path, param_model_path)

    logger.info("Getting model input ...")
    model_inputs = prep_model_inputs(
        agents_data,
        interaction_data.format(proc_member=model_id),
        target_data,
        cfg["train"]["interaction"],
        cfg["train"]["target"],
        cfg["train"]["interaction_ratio"],
    )

    logger.info("Building ABM ...")
    abm = build_abm(
        model_inputs["all_agents"],
        model_inputs["all_interactions"],
        cfg["train"]["infection"],
        cfg["predict"][proc_exp],
    )

    logger.info("Creating prediction ...")
    predictions = run_gradabm_wrapper(
        trained_output["param"]["param_with_smallest_loss"],
        trained_output["param_model"].param_info(),
        abm,
        trained_output["output_info"]["total_timesteps"],
        save_records=True,
    )

    logger.info("Output processing ...")
    output = postproc_pred(
        predictions,
        trained_output["output_info"]["target"],
        cfg["predict"]["common"]["start"]["timestep"],
        cfg["predict"]["common"]["end"]["timestep"],
    )
    write_output(
        output,
        trained_output["output_info"]["epoch_loss_list"],
        workdir,
        proc_exp,
        model_id,
        ens_id,
    )

    logger.info("Job done ...")


if __name__ == "__main__":
    args = setup_parser()
    main(
        args.workdir,
        args.cfg,
        args.model_base_dir,
        args.exp_id,
        args.model_id,
        args.ens_id,
    )
