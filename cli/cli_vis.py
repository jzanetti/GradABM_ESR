"""
Usage: cli_predict --workdir  /tmp/gradabm_esr_pred
                   --param_path /tmp/gradabm_esr/test1/params.p
                   --output_info_path /tmp/gradabm_esr/test1/output_info.p
                   --param_model_path /tmp/gradabm_esr/test1/param_model.model
                   --abm_model_path /tmp/gradabm_esr/test1/abm_model.model
Author: Sijin Zhang
Contact: sijin.zhang@esr.cri.nz

Description: 
    This is a wrapper to run the vis
"""

import argparse
from os.path import join

from input import RANDOM_ENSEMBLES, TRAINING_ENS_MEMBERS
from model.diags import plot_diags
from utils.utils import read_cfg


def get_example_usage():
    example_text = """example:
        * cli_vis

        """
    return example_text


def setup_parser():
    parser = argparse.ArgumentParser(
        description="Predicting GradABM model",
        epilog=get_example_usage(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--prd_dir",
        type=str,
        help="Prediction output dir path (e.g., /tmp/manukau_measles_2019)",
        required=True,
    )

    parser.add_argument(
        "--cfg",
        required=True,
        help="Configuration path for the model vis",
    )

    parser.add_argument(
        "--exp_name",
        required=True,
        help="Experiment name",
    )
    return parser.parse_args(
        [
            "--prd_dir",
            "exp/policy_paper/predict",
            "--cfg",
            "data/measles/policy_paper/base/vis_exp_vac1.yml",
            "--exp_name",
            "base_exp",
        ]
    )


def main(prd_dir, cfg, exp_name):
    from pickle import load as pickle_load

    cfg = read_cfg(cfg, key="vis")
    outputs = []
    epoch_losses = []
    for model_id in range(TRAINING_ENS_MEMBERS):
        for ens_id in range(RANDOM_ENSEMBLES):
            print(model_id, ens_id)
            proc_prd = pickle_load(
                open(join(prd_dir, exp_name, "output", f"pred_{model_id}_{ens_id}.p"), "rb")
            )
            outputs.append(proc_prd["output"])

            epoch_losses.append(proc_prd["epoch_loss_list"])

    plot_diags(
        join(prd_dir, exp_name),
        outputs,
        epoch_losses,
        cfg["vis"],
        cfg["timestep_cfg"],
        apply_log_for_loss=False,
        plot_obs=True,
    )


if __name__ == "__main__":
    args = setup_parser()
    main(args.prd_dir, args.cfg, args.exp_name)
