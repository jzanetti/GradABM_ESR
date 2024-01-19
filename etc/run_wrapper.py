from os.path import exists, join
from shutil import rmtree

from pandas import DataFrame

from cli.cli_input import main as input_main
from cli.cli_predict import main as predict_main
from cli.cli_train import main as train_main
from cli.cli_vis import main as vis_main
from input import RANDOM_ENSEMBLES, TRAINING_ENS_MEMBERS
from model.prep import get_prerun_params
from utils.utils import read_cfg

workdir = "exp/policy_paper"

input_data = {
    "june_nz_data": "data/measles/june_output",
    "cfg": "data/measles/base/input_exp_vac1.yml",
    "target_data": "data/measles_v2/base/measles_cases_2019.parquet",
    "sa2_dhb_data": "data/common/dhb_and_sa2.parquet",
    "dhb_list": ["Counties Manukau"],
}

model_cfg_path = "data/measles/base/gradam_exp_vac1.yml"
vis_cfg_path = "data/measles/base/vis_exp_vac1.yml"
prd_job_name = "base_exp"

run_input_main = True
run_prerun = False
run_model_main = True
run_predict_main = False
run_vis_main = False
remove_all_old_runs = False

if remove_all_old_runs:
    if exists(workdir):
        rmtree(workdir)

if run_input_main:
    input_main(
        join(workdir, "input"),
        input_data["cfg"],
        input_data["june_nz_data"],
        input_data["target_data"],
        input_data["dhb_list"],
        input_data["sa2_dhb_data"],
    )


if run_prerun:
    all_lost = {}
    all_params = get_prerun_params(model_cfg_path)

    all_keys = all_params[0].keys()
    all_lost = {key: [] for key in all_keys}
    all_lost["lost"] = []

    for i, proc_prep_param in enumerate(all_params):
        proc_pre_run_lost = train_main(
            join(workdir, "model"),
            "prerun",
            model_cfg_path,
            join(workdir, "input", "agents.parquet"),
            join(workdir, "input", f"interaction_graph_cfg_member_0.parquet"),
            join(workdir, "input", "output.csv"),
            prerun_params=proc_prep_param,
        )
        for proc_param_key in proc_prep_param:
            all_lost[proc_param_key].append(proc_prep_param[proc_param_key])
        all_lost["lost"].append(proc_pre_run_lost)

    DataFrame(all_lost).to_csv("prerun_stats.csv")
    print("Prerun: done")


if run_model_main:
    for proc_member in range(TRAINING_ENS_MEMBERS):
        train_main(
            join(workdir, "model"),
            f"member_{proc_member}",
            model_cfg_path,
            join(workdir, "input", "agents.parquet"),
            join(
                workdir, "input", f"interaction_graph_cfg_member_{proc_member}.parquet"
            ),
            join(workdir, "input", "output.csv"),
        )


if run_predict_main:
    for proc_member in range(TRAINING_ENS_MEMBERS):
        for ens_id in range(RANDOM_ENSEMBLES):
            print(f"Running pred: {proc_member}, {ens_id}")
            predict_main(
                join(workdir, "predict"),
                model_cfg_path,
                workdir,
                prd_job_name,
                proc_member,
                ens_id,
            )

if run_vis_main:
    print("Run vis ...")
    vis_main(join(workdir, "predict"), vis_cfg_path, prd_job_name)
