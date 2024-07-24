import warnings
from os.path import join

from process.input_wrapper import input_wrapper
from process.predict_wrapper import predict_wrapper
from process.train_wrapper import run_model_train_ens
from process.vis_wrapper import vis_wrapper

# Disable all warnings
warnings.filterwarnings("ignore")
"""
input_data = {
    "diary_path": "etc/tests/Auckland_2019_measles/raw_input_latest/Auckland/syspop_diaries.parquet",
    "synpop_path": "etc/tests/Auckland_2019_measles/raw_input_latest/Auckland/syspop_base.parquet",
    "target_data": "etc/tests/Auckland_2019_measles/raw_input_latest/Auckland/measles_cases_2019.parquet",
    "target_index_range": {"start": 25, "end": 51},
    "sa2_dhb_map_path": "etc/tests/Auckland_2019_measles/raw_input_latest/dhb_and_sa2.parquet",
    "dhb_list": [
        "Counties Manukau"
    ],  # Counties Manukau, Auckland, Capital and Coast, Canterbury
}

input_cfg_path = "etc/tests/Auckland_2019_measles/cfg/input.yml"
model_cfg_path = "etc/tests/papers/Manukau/cfg/model.yml"
vis_cfg_path = "etc/tests/papers/Manukau/cfg/vis.yml"
"""

input_data = {
    "diary_path": "etc/tests/Auckland_2019_measles/raw_input_v1.0/Auckland/diaries.parquet",
    "synpop_path": "etc/tests/Auckland_2019_measles/raw_input_v1.0/Auckland/syspop_base.parquet",
    "target_data": "etc/tests/Auckland_2019_measles/raw_input_v1.0/Auckland/measles_cases_2019.parquet",
    "target_index_range": {"start": 25, "end": 51},
    "sa2_dhb_map_path": "etc/tests/Auckland_2019_measles/raw_input_v1.0/Auckland/dhb_and_sa2.parquet",
    "dhb_list": [
        "Auckland"
    ],  # Counties Manukau, Auckland, Capital and Coast, Canterbury
}

input_cfg_path = "etc/tests/Auckland_2019_measles/cfg/input.yml"
model_cfg_path = "etc/tests/Auckland_2019_measles/cfg/model.yml"
vis_cfg_path = "etc/tests/papers/Manukau/cfg/vis.yml"

run_input = False
run_train = True
run_predict = False
run_vis = False

workdir = "etc/tests/debug/Auckland"

if run_input:
    input_wrapper(
        join(workdir, "input"),
        input_data["synpop_path"],
        input_data["diary_path"],
        input_cfg_path,
        target_path=input_data["target_data"],
        target_index_range=input_data["target_index_range"],
        geography_ancillary_path=input_data["sa2_dhb_map_path"],
        dhb_list=input_data["dhb_list"],
    )
if run_train:
    run_model_train_ens(join(workdir, "train"), model_cfg_path)

if run_predict:
    predict_wrapper(
        join(workdir, "predict"),
        model_cfg_path,
        max_ens=None,  # 15
        target_data_path=join(workdir, "input", "target.parquet"),
    )

if run_vis:
    vis_wrapper(join(workdir, "vis"), vis_cfg_path)
