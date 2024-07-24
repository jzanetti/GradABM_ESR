from logging import getLogger
from os import makedirs
from os.path import exists, join

from process.input.test import load_test_data
from process.model import OPTIMIZATION_CFG, PRINT_MODEL_INFO
from process.model.abm import init_abm
from process.model.diags import save_outputs
from process.model.loss_func import loss_optimization
from process.model.param import param_model_forward
from process.model.postp import postproc_train
from process.model.prep import get_train_all_paths, prep_wrapper
from process.model.wrapper import run_gradabm_wrapper
from process.utils.utils import (
    print_params_increments,
    print_prediction,
    read_cfg,
    setup_logging,
)


def run_model_train_ens(
    workdir: str,
    cfg_path: str,
):

    if not exists(workdir):
        makedirs(workdir)

    logger = setup_logging(workdir)

    all_paths = get_train_all_paths(join(workdir, ".."))

    ens_id = 0
    logger.info(f"Start model training...")

    total_ens = read_cfg(cfg_path, key="train")["ensembles"] * len(
        all_paths["interaction_paths"]
    )
    for _ in range(read_cfg(cfg_path, key="train")["ensembles"]):
        for proc_interaction_path in all_paths["interaction_paths"]:

            logger.info(f"Training the model: {ens_id} / {total_ens}")

            train(
                join(workdir, "model", f"member_{ens_id}"),
                cfg_path=cfg_path,
                agents_data_path=all_paths["agents_path"],
                interaction_data_path=proc_interaction_path,
                target_data_path=all_paths["target_path"],
            )
            ens_id += 1


def load_train_input(
    agents_data_path, interaction_data_path, target_data_path, cfg_path, use_test_data
):
    if use_test_data:
        train_input = load_test_data(large_network=True)
    else:
        train_input = prep_wrapper(
            agents_data_path, interaction_data_path, target_data_path, cfg_path
        )

    return train_input["cfg"], train_input["model_inputs"]


def train(
    workdir: str,
    cfg_path: str or None = None,
    agents_data_path: str or None = None,
    interaction_data_path: str or None = None,
    target_data_path: str or None = None,
    use_test_data: bool = False,
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
    cfg, model_inputs = load_train_input(
        agents_data_path,
        interaction_data_path,
        target_data_path,
        cfg_path,
        use_test_data,
    )

    logger.info("    * Building and running ABM ...")
    abm = init_abm(model_inputs, cfg)

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
            f"    * step {epi}: "
            f"Loss: {round(epoch_loss, 2)}/{round(smallest_loss, 2)}; "
            f"Lr: {round(abm['loss_def']['opt'].param_groups[0]['lr'], 5)}"
        )
        print_prediction(output["pred"], output["y"])

        if PRINT_MODEL_INFO:
            print_params_increments(param_values_list)

        if epoch_loss < smallest_loss:
            param_with_smallest_loss = param_values_all
            smallest_loss = epoch_loss
        epoch_loss_list.append(epoch_loss)
        param_values_list.append(param_values_all)

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
