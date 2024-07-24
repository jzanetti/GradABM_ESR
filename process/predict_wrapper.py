from logging import getLogger
from os import makedirs
from os.path import exists, join

from numpy.random import choice as numpy_choice
from pandas import read_parquet as pandas_read_parquet

from process.model.abm import build_abm
from process.model.diags import load_outputs
from process.model.postp import postproc_pred, write_output
from process.model.prep import get_all_pred_pahts, prep_model_inputs
from process.model.wrapper import run_gradabm_wrapper
from process.utils.utils import read_cfg, setup_logging


def produce_single_prediction(
    workdir: str,
    ens_id: str,
    cfg: dict,
    proc_agents_path: str,
    proc_interaction_path: str,
    trained_output: dict,
    output_prefix: str or None,
):

    logger = getLogger()
    logger.info("Getting model input ...")
    model_inputs = prep_model_inputs(
        proc_agents_path,
        proc_interaction_path,
        None,
        cfg["train"]["interaction"],
    )

    abm = build_abm(
        model_inputs["all_agents"],
        model_inputs["all_interactions"],
        cfg["train"]["infection"],
        cfg["train"]["outbreak_ctl"],
        cfg_update=cfg["predict"]["updated_cfg"],
    )

    logger.info("Creating prediction ...")
    predictions = run_gradabm_wrapper(
        abm,
        trained_output["param"]["param_with_smallest_loss"],
        trained_output["param_model"].param_info(),
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
        ens_id,
        output_prefix,
    )


def predict_wrapper(
    workdir: str,
    cfg: str,
    max_ens: int = None,
    target_data_path: str = None,
    output_prefix: str or None = None,
    agent_data_path: str = None,
):
    """Createing prediction based on previously trained model

    Args:
        workdir (str): Prediction working directoty
        cfg (str): configuration
        max_ens (int, optional): _description_. Defaults to None.
    """
    if not exists(workdir):
        makedirs(workdir)

    logger = setup_logging(workdir)

    logger.info("Reading configuration ...")
    if isinstance(cfg, str):
        cfg_list = [read_cfg(cfg)]
    elif isinstance(cfg, list):
        cfg_list = []
        for proc_cfg in cfg:
            cfg_list.append(read_cfg(proc_cfg))

    all_pred_paths = get_all_pred_pahts(workdir, agent_data_path)

    ens_id = 0

    for proc_trained_models_dir in all_pred_paths["all_trained_models_dirs"]:

        param_path = join(proc_trained_models_dir, "params.p")
        output_info_path = join(proc_trained_models_dir, "output_info.p")
        param_model_path = join(proc_trained_models_dir, "param_model.model")

        trained_output = load_outputs(param_path, output_info_path, param_model_path)

        if target_data_path is not None:
            target_data = pandas_read_parquet(target_data_path).to_numpy()
            trained_output["output_info"]["target"] = target_data.reshape(
                1, len(target_data), 1
            )

        for proc_interaction_path in all_pred_paths["all_interactions_paths"]:

            if max_ens is not None and ens_id > max_ens:
                logger.info(f"Reached the max prediction limit: {max_ens}, Quit ...")
                return

            logger.info(
                f"Creating predictions: {ens_id} / {all_pred_paths['total_ens']}"
            )
            produce_single_prediction(
                workdir,
                ens_id,
                numpy_choice(cfg_list),
                all_pred_paths["agents_path"],
                proc_interaction_path,
                trained_output,
                output_prefix,
            )
            ens_id += 1

    logger.info("Job done ...")
