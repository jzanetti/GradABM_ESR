"""
Usage: cli_june --workdir /tmp/june_nz --cfg june.cfg
Author: Sijin Zhang
Contact: sijin.zhang@esr.cri.nz

Description: 
    This is a wrapper to run the JUNE model
"""

import argparse
from os import makedirs
from os.path import exists, join

from model import INITIAL_LOSS, NUM_EPOCHS
from model.abm import build_abm, forward_abm
from model.diags import save_outputs
from model.loss_func import get_loss_func, loss_optimization
from model.param_model import create_param_model, param_model_forward
from model.postp import postproc
from model.prep import prep_env, prep_model_inputs
from utils.utils import read_cfg, setup_logging


def get_example_usage():
    example_text = """example:
        * cli_june --workdir /tmp/june_nz
                    --cfg june.cfg
        """
    return example_text


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Run GradABM model",
        epilog=get_example_usage(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--exp", required=True, help="Experiment name, e.g., exp1")
    parser.add_argument(
        "--workdir", required=True, help="Working directory, e.g., where the output will be stored"
    )
    parser.add_argument(
        "--cfg",
        required=True,
        help="Configuration path for the model (e.g., scaling factor), e.g., gradadm_exp.cfg",
    )
    parser.add_argument(
        "--learnable_param", required=True, help="Learnable paramters configuration"
    )
    parser.add_argument("--agents_data", required=True, help="Agents data in parquet")
    parser.add_argument("--interaction_data", required=True, help="Interaction data in parquet")
    parser.add_argument("--target_data", required=True, help="Target data in CSV")
    return parser.parse_args(
        [
            "--exp",
            "test1",
            "--workdir",
            "/tmp/gradabm_esr",
            "--cfg",
            "cfg/sample_cfg/gradam_exp.yml",
            "--learnable_param",
            "cfg/sample_cfg/learnable_param.yml",
            "--agents_data",
            "/tmp/gradabm_esr_input/agents.parquet",
            "--interaction_data",
            "/tmp/gradabm_esr_input/interaction_graph_cfg.parquet",
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

    logger.info("Reading configuration ...")
    cfg = read_cfg(args.cfg)

    logger.info("Preparing model running environment ...")
    prep_env()

    logger.info("Getting model input ...")
    model_inputs = prep_model_inputs(
        args.agents_data, args.interaction_data, args.target_data, cfg["interaction"]
    )

    logger.info("Building ABM ...")
    abm = build_abm(model_inputs["all_agents"], model_inputs["all_interactions"], cfg["infection"])

    logger.info("Creating initial parameters (to be trained) ...")
    param_model = create_param_model(args.learnable_param)

    logger.info("Creating loss function ...")
    loss_def = get_loss_func(param_model, model_inputs["total_timesteps"])

    epoch_loss_list = []
    smallest_loss = INITIAL_LOSS

    for epi in range(NUM_EPOCHS):
        param_values_all = param_model_forward(param_model, model_inputs["target"])

        predictions = forward_abm(
            param_values_all,
            param_model.param_info(),
            abm,
            model_inputs["total_timesteps"],
            save_records=False,
        )

        output = postproc(param_model, predictions, model_inputs["target"])

        loss = loss_def["loss_func"](output["y"], output["pred"])

        epoch_loss = loss_optimization(loss, param_model, loss_def)

        logger.info(
            f"{epi}: Loss: {round(epoch_loss, 2)}/{round(smallest_loss, 2)}; Lr: {round(loss_def['opt'].param_groups[0]['lr'], 2)}"
        )

        if epoch_loss < smallest_loss:
            param_with_smallest_loss = param_values_all
            smallest_loss = epoch_loss

        epoch_loss_list.append(epoch_loss)

    logger.info(param_values_all)

    logger.info("Save trained model ...")

    save_outputs(
        {
            "output_info": {
                "total_timesteps": model_inputs["total_timesteps"],
                "target": model_inputs["target"],
                "epoch_loss_list": epoch_loss_list,
            },
            "params": {"param_with_smallest_loss": param_with_smallest_loss},
            "param_model": param_model,
        },
        join(args.workdir, args.exp),
    )

    logger.info("Job done ...")


if __name__ == "__main__":
    main()
