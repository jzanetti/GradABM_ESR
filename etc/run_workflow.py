import warnings
from os.path import join

from process.input_wrapper import input_wrapper
from process.predict_wrapper import predict_wrapper
from process.train_wrapper import train_wrapper
from process.vis_wrapper import vis_wrapper

# Disable all warnings
warnings.filterwarnings("ignore")

input_data = {
    "diary_path": "etc/tests/Auckland_2019_measles/raw_input/diaries.parquet",
    "synpop_path": "etc/tests/Auckland_2019_measles/raw_input/syspop_base.parquet",
    "cfg": "etc/tests/Auckland_2019_measles/cfg/input.yml",
    "target_data": "etc/tests/Auckland_2019_measles/raw_input/measles_cases_2019.parquet",
    "target_index_range": {"start": 25, "end": 51},
    "sa2_dhb_map_path": "etc/tests/Auckland_2019_measles/raw_input/dhb_and_sa2.parquet",
    "dhb_list": ["Counties Manukau"],
}

model_cfg_path = "etc/tests/Auckland_2019_measles/cfg/model.yml"
vis_cfg_path = "etc/tests/Auckland_2019_measles/cfg/vis.yml"

run_input = False
run_train = True
run_predict = True
run_vis = True

workdir = "/tmp/gradabm_esr/Auckland_2019_measles3"

if run_input:
    input_wrapper(
        join(workdir, "input"),
        input_data["synpop_path"],
        input_data["diary_path"],
        input_data["cfg"],
        target_path=input_data["target_data"],
        target_index_range=input_data["target_index_range"],
        geography_ancillary_path=input_data["sa2_dhb_map_path"],
        dhb_list=input_data["dhb_list"],
    )
if run_train:
    train_wrapper(
        join(workdir, "train"), model_cfg_path, run_prerun=True, use_prerun=True
    )

if run_predict:
    predict_wrapper(join(workdir, "predict"), model_cfg_path, max_ens=None)

if run_vis:
    vis_wrapper(join(workdir, "vis"), vis_cfg_path)
