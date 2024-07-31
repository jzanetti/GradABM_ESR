from numpy import array as numpy_array
from pandas import DataFrame
from torch import Tensor as torch_tensor

from process import DEVICE
from process.model import OPTIMIZATION_CFG
from process.model.abm import GradABM


def run_gradabm_wrapper(
    abm: GradABM,
    param_values_all: torch_tensor,
    param_info: dict,
    training_num_steps: int,
    save_records: bool = False,
):
    """Run GradABM ESR wrapper

    Args:
        abm (GradABM): GradABM model
        param_values_all (torch_tensor): _description_
        param_info (dict): _description_
        training_num_steps (int): _description_
        save_records (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    predictions_indices = []
    all_records = []

    param_values_all = param_values_all.to(DEVICE)

    pred = 0
    for time_step in range(training_num_steps):

        if OPTIMIZATION_CFG["use_temporal_params"]:
            param_values = param_values_all[0, time_step, :].to(DEVICE)
        else:
            param_values = param_values_all

        proc_record, pred, pred_indices = abm.step(
            time_step,
            param_values,
            param_info,
            training_num_steps,
            pred,
            save_records=save_records,
        )

        predictions = pred.to(DEVICE)
        predictions_indices.append(pred_indices)
        all_records.append(proc_record)

    predictions = predictions[-1]

    if any(item is None for item in all_records):
        output = None
    else:
        output = numpy_array(all_records).T
        output = DataFrame(output).astype(int)
        output["area"] = abm.agents_area.tolist()
        output["ethnicity"] = abm.agents_ethnicity.tolist()
        output["age"] = abm.agents_age.tolist()
        output["vaccine"] = abm.agents_vaccine.tolist()
        output["gender"] = abm.agents_gender.tolist()

    return {
        "prediction": predictions,
        "prediction_indices": predictions_indices,
        "output": output,
    }
