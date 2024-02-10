from logging import getLogger
from os import makedirs
from os.path import exists, join
from random import sample as random_sample

from pandas import DataFrame

from process.model import OPTIMIZATION_CFG, PRINT_MODEL_INFO
from process.model.abm import init_abm
from process.model.diags import save_outputs
from process.model.loss_func import loss_optimization
from process.model.param import param_model_forward
from process.model.postp import postproc_train
from process.model.prep import (
    get_prerun_params,
    get_train_all_paths,
    prep_wrapper,
    update_train_cfg_using_prerun,
)
from process.model.wrapper import run_gradabm_wrapper
from process.utils.utils import print_params_increments, read_cfg, setup_logging


def train_wrapper(
    workdir: str, cfg_path: str, run_prerun: bool = False, use_prerun: bool = False
):

    if not exists(workdir):
        makedirs(workdir)

    logger = setup_logging(workdir)

    all_paths = get_train_all_paths(join(workdir, ".."))

    # ----------------------------------------------
    # Step 1: Run prerun (sensitivity studies)
    # ----------------------------------------------
    if run_prerun:
        all_lost = {}
        all_params = get_prerun_params(read_cfg(cfg_path, key="train"))

        all_keys = all_params[0].keys()
        all_lost = {key: [] for key in all_keys}
        all_lost["lost"] = []

        logger.info(f"All pre-run experiments to run: {len(all_params)} ...")

        for i, proc_prep_param in enumerate(all_params):

            logger.info(f"   - start prerun exp {i}/{len(all_params)}")

            proc_pre_run_lost = run_model_train(
                join(workdir, "prerun"),
                cfg_path,
                all_paths["agents_path"],
                all_paths["interaction_paths"][0],
                all_paths["target_path"],
                prerun_params=proc_prep_param,
            )
            for proc_param_key in proc_prep_param:
                all_lost[proc_param_key].append(proc_prep_param[proc_param_key])
            all_lost["lost"].append(proc_pre_run_lost)

        DataFrame(all_lost).sort_values(by="lost").to_csv(
            join(workdir, "prerun", "prerun_stats.csv"), index=False
        )
        logger.info("Prerun: job completed")

    # ----------------------------------------------
    # Step 2: Updated configuration file using the pre-run stats
    # ----------------------------------------------
    if use_prerun:
        logger.info(f"Updated configuration file based on prerun...")
        updated_cfg_paths = update_train_cfg_using_prerun(
            join(workdir, "updated_cfg"), cfg_path
        )
    else:
        updated_cfg_paths = [cfg_path]

    # ----------------------------------------------
    # Step 3: Run the model training
    # ----------------------------------------------
    ens_id = 0
    logger.info(f"Start model training...")
    for _ in range(read_cfg(cfg_path, key="train")["ensembles"]):
        for proc_interaction_path in all_paths["interaction_paths"]:
            run_model_train(
                join(workdir, "model", f"member_{ens_id}"),
                random_sample(updated_cfg_paths, 1)[0],
                all_paths["agents_path"],
                proc_interaction_path,
                all_paths["target_path"],
            )
            ens_id += 1


def run_model_train(
    workdir: str,
    cfg_path: str,
    agents_data_path: str,
    interaction_data_path: str,
    target_data_path: str,
    prerun_params: list or None = None,  # type: ignore
):
    """Run the model training for GradABM_ESR

    Args:
        workdir (str): Working directory
        cfg_path (str): Configuration path
        agents_data_path (str): Agent data path
        interaction_data_path (str): Interaction data path
        target_data_path (str): Target data path
        prerun_params (list or None, optional): Prerun parameters. Defaults to None.
    """

    if not exists(workdir):
        makedirs(workdir)

    logger = getLogger()

    logger.info("    * Preprocessing ...")
    model_inputs, cfg = prep_wrapper(
        agents_data_path, interaction_data_path, target_data_path, cfg_path
    )

    logger.info("    * Building and running ABM ...")
    abm = init_abm(model_inputs, cfg, prerun_params)

    epoch_loss_list = []
    param_values_list = []
    smallest_loss = OPTIMIZATION_CFG["initial_loss"]
    for epi in range(abm["num_epochs"]):
        param_values_all = param_model_forward(
            abm["param_model"], model_inputs["target"]
        )
        predictions = run_gradabm_wrapper(
            abm["model"],
            param_values_all,
            abm["param_model"].param_info(),
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
            f"       * step {epi}: "
            f"Loss: {round(epoch_loss, 2)}/{round(smallest_loss, 2)}; "
            f"Lr: {round(abm['loss_def']['opt'].param_groups[0]['lr'], 5)}"
        )

        if PRINT_MODEL_INFO:
            print_params_increments(param_values_list)

        if epoch_loss < smallest_loss:
            param_with_smallest_loss = param_values_all
            smallest_loss = epoch_loss
        epoch_loss_list.append(epoch_loss)
        param_values_list.append(param_values_all)

    if prerun_params:
        logger.info("    * Saved smallest loss for the prerun ...")
        return smallest_loss

    # logger.info(param_values_all)

    logger.info("    * Save trained model ...")

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
        workdir,
    )

    logger.info("    * Job done ...")
