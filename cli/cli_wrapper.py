from os.path import join

from cli.cli_input import main as input_main
from cli.cli_predict import main as predict_main
from cli.cli_train import main as train_main
from input import RANDOM_ENSEMBLES, TRAINING_ENS_MEMBERS

workdir = "exp/policy_paper2"

input_data = {
    "june_nz_data": "data/measles/june_output",
    "cfg": "data/measles/base/input_exp_vac1.yml",
    "target_data": "data/measles/base/measles_cases_2019.parquet",
    "sa2_dhb_data": "data/common/dhb_and_sa2.parquet",
    "dhb_list": ["Counties Manukau"],
}

model_cfg_path = "data/measles/base/gradam_exp_vac1.yml"
prd_job_name = "base_exp"
run_input_main = True
run_model_main = True
run_predict_main = True


if run_input_main:
    print(f"Creating input data ...")
    input_main(
        join(workdir, "input"),
        input_data["cfg"],
        input_data["june_nz_data"],
        input_data["target_data"],
        input_data["dhb_list"],
        input_data["sa2_dhb_data"],
    )

if run_model_main:
    for proc_member in range(TRAINING_ENS_MEMBERS):
        print(f"Running train: {proc_member}")
        train_main(
            join(workdir, "model"),
            f"member_{proc_member}",
            model_cfg_path,
            join(workdir, "input", "agents.parquet"),
            join(workdir, "input", f"interaction_graph_cfg_member_{proc_member}.parquet"),
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
