from os.path import join

from cli.cli_input import main as input_main
from cli.cli_predict import main as predict_main
from cli.cli_train import main as train_main
from input import INTERACTION_ENS_MEMBERS

workdir = "/tmp/manukau_measles_2019/policy_paper"

input_data = {
    "june_nz_data": "data/june_output/interaction_output.parquet",
    "cfg": "data/measles/policy_paper/base/input_exp_vac1.yml",
    "target_data": "data/measles_cases_2019.parquet",
    "sa2_dhb_data": "data/dhb_and_sa2.parquet",
    "dhb_list": ["Counties Manukau"],
}

model_cfg_path = "data/measles/policy_paper/base/gradam_exp_vac1.yml"
run_input_main = False
run_model_main = False
run_predict_main = True

if run_input_main:
    input_main(
        join(workdir, "input"),
        input_data["cfg"],
        input_data["june_nz_data"],
        input_data["target_data"],
        input_data["dhb_list"],
        input_data["sa2_dhb_data"],
    )

if run_model_main:
    for proc_member in range(INTERACTION_ENS_MEMBERS):
        train_main(
            join(workdir, "model"),
            f"member_{proc_member}",
            model_cfg_path,
            join(workdir, "input", "agents.parquet"),
            join(workdir, "input", f"interaction_graph_cfg_member_{proc_member}.parquet"),
            join(workdir, "input", "output.csv"),
        )


if run_predict_main:
    predict_main(join(workdir, "predict"), model_cfg_path, workdir)
