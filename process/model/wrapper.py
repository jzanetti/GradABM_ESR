import torch
from numpy import array as numpy_array
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
    predictions = []
    all_records = []
    all_target_indices = []

    param_values_all = param_values_all.to(DEVICE)

    for time_step in range(training_num_steps):
        if OPTIMIZATION_CFG["use_temporal_params"]:
            param_values = param_values_all[0, time_step, :].to(DEVICE)
        else:
            param_values = param_values_all

        proc_record, target_indices, pred_t = abm.step(
            time_step,
            param_values,
            param_info,
            training_num_steps,
            save_records=save_records,
        )
        pred_t = pred_t.type(torch.float64)
        predictions.append(pred_t.to(DEVICE))
        all_records.append(proc_record)
        all_target_indices.append(target_indices)

    predictions = torch.stack(predictions, 0).reshape(1, -1)

    if any(item is None for item in all_records):
        all_records = None
    else:
        all_records = numpy_array(all_records)

    return {
        "prediction": predictions,
        "all_records": all_records,
        "all_target_indices": all_target_indices,
        "agents_area": abm.agents_area.tolist(),
        "agents_ethnicity": abm.agents_ethnicity.tolist(),
    }
