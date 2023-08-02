from os.path import join

from cli.cli_input import main as input_main
from cli.cli_predict import main as predict_main
from cli.cli_train import main as train_main
from input import INTERACTION_ENS_MEMBERS

workdir = "/tmp/manukau_measles_2019"

input_data = {
    "june_nz_data": "data/june_output/interaction_output.parquet",
    "cfg": "data/measles/auckland/input_exp.yml",
    "target_data": "data/measles_cases_2019.parquet",
    "sa2_dhb_data": "data/dhb_and_sa2.parquet",
    "dhb_list": ["Counties Manukau"],
}

model_predict_cfg = "data/measles/auckland/gradam_exp.yml"
run_input_main = True
run_model_main = True
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
            model_predict_cfg,
            join(workdir, "input", "agents.parquet"),
            join(workdir, "input", f"interaction_graph_cfg_member_{proc_member}.parquet"),
            join(workdir, "input", "output.csv"),
        )


if run_predict_main:
    predict_main(join(workdir, "predict"), model_predict_cfg, workdir, replace_agents_with=None)
