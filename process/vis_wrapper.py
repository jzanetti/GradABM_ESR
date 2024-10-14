from glob import glob
from os import makedirs
from os.path import exists, join
from pickle import load as pickle_load

from process.model.diags import plot_diags
from process.utils.utils import read_cfg


def vis_wrapper(workdir: str, cfg: str):
    cfg = read_cfg(cfg, key="vis")
    outputs = []
    epoch_losses = []

    if not exists(workdir):
        makedirs(workdir)

    all_prd_paths = glob(join(workdir, "..", "predict", "pred_*.pickle"))
    for proc_prd_path in all_prd_paths:
        proc_prd = pickle_load(
            open(
                proc_prd_path,
                "rb",
            )
        )
        outputs.append(proc_prd["output"])
        epoch_losses.append(proc_prd["epoch_loss_list"])

    plot_diags(
        workdir,
        outputs,
        epoch_losses,
        cfg["vis"],
        cfg["timestep_cfg"],
        apply_log_for_loss=False,
        plot_obs=True,
    )

    
