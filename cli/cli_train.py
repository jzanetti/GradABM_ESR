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

from process.model import INITIAL_LOSS, PRERUN_NUM_EPOCHS, PRINT_INCRE
from process.model.abm import build_abm, init_abm
from process.model.diags import save_outputs
from process.model.loss_func import get_loss_func, loss_optimization
from process.model.param import build_param_model, obtain_param_cfg, param_model_forward
from process.model.postp import postproc_train
from process.model.prep import (
    prep_env,
    prep_model_inputs,
    prep_wrapper,
    update_params_for_prerun,
)
from process.model.wrapper import run_gradabm_wrapper
from utils.utils import get_params_increments, read_cfg, setup_logging

prep_wrapper


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
        "--workdir",
        required=True,
        help="Working directory, e.g., where the output will be stored",
    )
    parser.add_argument(
        "--cfg",
        required=True,
        help="Configuration path for the model (e.g., scaling factor), e.g., gradadm_exp.cfg",
    )
    parser.add_argument("--agents_data", required=True, help="Agents data in parquet")
    parser.add_argument(
        "--interaction_data", required=True, help="Interaction data in parquet"
    )
    parser.add_argument("--target_data", required=True, help="Target data in CSV")

    """
    return parser.parse_args(
        [
            "--exp",
            "test1",
            "--workdir",
            "/tmp/gradabm_esr_auckland",
            "--cfg",
            "data/measles/auckland/gradam_exp_vac1.yml",
            # "--learnable_param",
            # "data/measles/auckland/learnable_param.yml",
            "--agents_data",
            "data/measles/auckland/inputs/agents.parquet",
            "--interaction_data",
            "data/measles/auckland/inputs/interaction_graph_cfg_member_0.parquet",
            "--target_data",
            "data/measles/auckland/inputs/output.csv",
        ]
    )
    """
    return parser.parse_args()


def main(
    workdir: str,
    exp_name: str,
    cfg_path: str,
    agents_data_path: str,
    interaction_data_path: str,
    target_data_path: str,
    prerun_params: list or None = None,
):
    """Run the model training for GradABM_ESR

    Args:
        workdir (str): Working directory
        exp_name (str): Experimental name
        cfg_path (str): Configuration path
        agents_data_path (str): Agent data path
        interaction_data_path (str): Interaction data path
        target_data_path (str): Target data path
        prerun_params (list or None, optional): Prerun parameters. Defaults to None.
    """

    if not exists(workdir):
        makedirs(workdir)

    logger = setup_logging(workdir)

    logger.info("Preprocessing ...")
    model_inputs, cfg = prep_wrapper(
        agents_data_path, interaction_data_path, target_data_path, cfg_path
    )

    logger.info("Building ABM ...")
    abm = init_abm(model_inputs, cfg, prerun_params)

    epoch_loss_list = []
    param_values_list = []
    smallest_loss = INITIAL_LOSS
    for epi in range(abm["num_epochs"]):
        param_values_all = param_model_forward(
            abm["param_model"], model_inputs["target"]
        )
        predictions = run_gradabm_wrapper(
            param_values_all,
            abm["param_model"].param_info(),
            abm["model"],
            model_inputs["total_timesteps"],
            save_records=False,
        )

        output = postproc_train(predictions, model_inputs["target"])

        loss = abm["loss_def"]["loss_func"](output["y"], output["pred"])

        epoch_loss = loss_optimization(
            loss,
            abm["param_model"],
            abm["loss_def"],
            print_grad=False,
        )

        logger.info(
            f"{epi}: Loss: {round(epoch_loss, 2)}/{round(smallest_loss, 2)}; Lr: {round(abm['loss_def']['opt'].param_groups[0]['lr'], 5)}"
        )

        if PRINT_INCRE:
            get_params_increments(param_values_list)

        if epoch_loss < smallest_loss:
            param_with_smallest_loss = param_values_all
            smallest_loss = epoch_loss
        epoch_loss_list.append(epoch_loss)
        param_values_list.append(param_values_all)

    if prerun_params:
        logger.info("Saved smallest loss for the prerun ...")
        return smallest_loss

    logger.info(param_values_all)

    logger.info("Save trained model ...")

    save_outputs(
        {
            "output_info": {
                "total_timesteps": model_inputs["total_timesteps"],
                "target": model_inputs["target"],
                "epoch_loss_list": epoch_loss_list,
                "param_values_list": param_values_list,
            },
            "all_interactions": model_inputs["all_interactions"],
            "params": {"param_with_smallest_loss": param_with_smallest_loss},
            "param_model": abm["param_model"],
        },
        join(workdir, exp_name),
    )

    logger.info("Job done ...")


if __name__ == "__main__":
    args = setup_parser()
    main(
        args.workdir,
        args.exp,
        args.cfg,
        args.agents_data,
        args.interaction_data,
        args.target_data,
    )
