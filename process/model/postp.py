from os import makedirs
from os.path import exists, join
from pickle import dump as pickle_dump

from process.model import REMOVE_WARM_UP_TIMESTEPS


def write_output(output, epoch_loss_list, workdir, ens_id):
    output_name = f"pred_{ens_id}.pickle"
    output_path = join(workdir, output_name)

    pickle_dump(
        {
            "output": output,
            "epoch_loss_list": epoch_loss_list,
        },
        open(output_path, "wb"),
    )


def postproc_train(prediction, y) -> dict:
    """Warm-up period must be excluded: warm-up period is usually
        equals to infected_to_recovered_or_death_time, we need to remove it since
        it is not generated by the model (e.g., people gradually die over time). In contrast,
        the death after warm-up will suddenly emerge from the initial infection.

    Args:
        param_model (_type_): _description_
        prediction (_type_): _description_
        y (_type_): _description_
    """
    output = {
        "pred": prediction["prediction"][0, :],
        "all_records": prediction["all_records"],
        "y": y[0, :, 0],
        "all_target_indices": prediction["all_target_indices"],
        "agents_area": prediction["agents_area"],
        "agents_ethnicity": prediction["agents_ethnicity"],
    }
    if REMOVE_WARM_UP_TIMESTEPS is not None:
        output["pred"] = output["pred"][REMOVE_WARM_UP_TIMESTEPS:]

        if output["all_records"] is not None:
            output["all_records"] = output["all_records"][REMOVE_WARM_UP_TIMESTEPS:]

        output["y"] = output["y"][REMOVE_WARM_UP_TIMESTEPS:]

    return output


def postproc_pred(prediction, y, start_t, end_t) -> dict:
    """Warm-up period must be excluded: warm-up period is usually
        equals to infected_to_recovered_or_death_time, we need to remove it since
        it is not generated by the model (e.g., people gradually die over time). In contrast,
        the death after warm-up will suddenly emerge from the initial infection.

    Args:
        param_model (_type_): _description_
        prediction (_type_): _description_
        y (_type_): _description_
    """
    try:
        obs_data = y[0, :, 0].detach().cpu().numpy()
    except AttributeError:
        obs_data = y[0, :, 0]

    output = {
        "obs": obs_data,
        "pred": prediction["prediction"][0, :].detach().cpu().numpy(),
        "stages": {
            "all_records": prediction["all_records"],
            "all_indices": prediction["all_target_indices"],
        },
        "agents": {
            "area": prediction["agents_area"],
            "ethnicity": prediction["agents_ethnicity"],
            "gender": prediction["agents_gender"],
            "age": prediction["agents_age"],
            "vaccine": prediction["agents_vaccine"],
        },
    }
    if start_t is None:
        start_t = 0
    if end_t is None:
        end_t = -1

    output["pred"] = output["pred"][start_t:end_t]

    if output["stages"]["all_records"] is not None:
        output["stages"]["all_records"] = output["stages"]["all_records"][start_t:end_t]

    output["obs"] = output["obs"][start_t:end_t]

    return output
