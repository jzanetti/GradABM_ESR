from copy import copy as shallow_copy
from json import load as json_load
from logging import getLogger

import torch
import torch.nn.functional as torch_func
from numpy import isnan as numpy_isnan
from pandas import DataFrame
from pandas import read_csv as pandas_read_csv
from torch import manual_seed
from torch.distributions import Gamma as torch_gamma
from torch_geometric.data import Data

from process import DEVICE
from process.model import (
    ALL_PARAMS,
    OPTIMIZATION_CFG,
    PERTURBATE_FLAG_DEFAULT,
    PRERUN_CFG,
    PRINT_MODEL_INFO,
    STAGE_INDEX,
    TORCH_SEED,
    USE_RANDOM_EXPOSED_DEFAULT,
)
from process.model.gnn import GNN_model
from process.model.loss_func import get_loss_func
from process.model.param import build_param_model
from process.model.prep import get_learnable_params_scaler
from process.model.progression import Progression_model
from process.model.utils import apply_gumbel_softmax

logger = getLogger()


class GradABM:
    def __init__(self, params):
        # -----------------------------
        # Step 1: set up agents
        # -----------------------------
        for attr in ["id", "age", "gender", "ethnicity", "vaccine", "area"]:
            setattr(
                self,
                f"agents_{attr}",
                torch.tensor(params["all_agents"][attr].to_numpy()).long().to(DEVICE),
            )
        self.num_agents = len(params["all_agents"])

        # -----------------------------
        # Step 2: set up interaction
        # -----------------------------
        self.all_edgelist = params["all_interactions"]["all_edgelist"]
        self.all_edgeattr = params["all_interactions"]["all_edgeattr"]

        # -----------------------------
        # Step 3: set up prediction configuration
        # -----------------------------
        self.initial_infected_ids = params["predict_update"][
            "initial_infected_ids_update"
        ]
        self.use_random_exposed = USE_RANDOM_EXPOSED_DEFAULT
        if params["predict_update"]["use_random_exposed_update"] is not None:
            self.use_random_exposed = params["predict_update"][
                "use_random_exposed_update"
            ]

        self.outbreak_ctl_cfg = params["outbreak_ctl_cfg"]
        if params["predict_update"]["outbreak_ctl_cfg_update"] is not None:
            self.outbreak_ctl_cfg = params["predict_update"]["outbreak_ctl_cfg_update"]

        self.perturbation_flag = PERTURBATE_FLAG_DEFAULT
        if params["predict_update"]["perturbation_flag_update"] is not None:
            self.perturbation_flag = params["predict_update"][
                "perturbation_flag_update"
            ]

        # -----------------------------
        # Step 5: set up scaling_factor
        # -----------------------------
        for attr in ["age", "symptom", "ethnicity", "gender", "vaccine"]:
            if attr in ["vaccine", "ethnicity"]:
                try:
                    proc_attr_cfg = params["predict_update"]["scaling_factor_update"][
                        attr
                    ]
                except (KeyError, TypeError):
                    proc_attr_cfg = params["infection_cfg"]["scaling_factor"][
                        f"{attr}_dependant"
                    ]
                params["infection_cfg"]["scaling_factor"][
                    f"{attr}_dependant"
                ] = proc_attr_cfg

            setattr(
                self,
                f"scaling_factor_{attr}",
                torch.tensor(
                    params["infection_cfg"]["scaling_factor"][f"{attr}_dependant"]
                )
                .float()
                .to(DEVICE),
            )

        # -----------------------------
        # Step 6: set up scaling_factor
        # -----------------------------
        self.learnable_params_scaler_update = params["predict_update"][
            "learnable_params_scaler_update"
        ]

        # -----------------------------
        # Step 7: set up progress and GNN model
        # -----------------------------
        self.progression_model = Progression_model(self.num_agents)

        self.gnn_model = GNN_model(
            self.scaling_factor_age,
            self.scaling_factor_gender,
            self.scaling_factor_ethnicity,
            self.scaling_factor_vaccine,
            self.scaling_factor_symptom,
            # self.lam_gamma_integrals,
            # self.device,
        ).to(DEVICE)

        self.current_time = 0

    def get_newly_exposed(
        self,
        lam_gamma_integrals,
        vaccine_efficiency_spread,
        contact_tracing_coverage,
        t,
        total_timesteps,
    ):

        # --------------------------------------------
        # Step 1: This step is to make sure that edgelist[1, :] starts from 0 to n ~
        # it should has the same sequence/order as the self.current_stage etc.
        #    Reason: edgelist[1, :] is considered the source node,
        #            self.gnn_model will write the output at the source node
        # --------------------------------------------
        _, indices = torch.sort(self.all_edgelist[1, :])
        sorted_all_edgelist = self.all_edgelist[:, indices]
        sorted_all_edgeattr = self.all_edgeattr[:, indices]

        # --------------------------------------------
        # Step 2: Start creating the infectious weighting
        # --------------------------------------------
        all_nodeattr = torch.stack(
            (
                self.agents_id,  # 0: id
                self.agents_age,  # 1: age
                self.agents_gender,  # 2: sex
                self.agents_ethnicity,  # 3: ethnicity
                self.agents_vaccine,  # 4: vaccine
                self.current_stages.detach(),  # 5: stage
                self.agents_infected_index.to(DEVICE),  # 6: infected index
                self.agents_infected_time.to(DEVICE),  # 7: infected time
            )
        ).t()

        agents_data = Data(
            all_nodeattr,
            edge_index=sorted_all_edgelist,
            edge_attr=sorted_all_edgeattr,
            vaccine_efficiency_spread=vaccine_efficiency_spread,
            contact_tracing_coverage=contact_tracing_coverage,
            t=t,
            total_timesteps=total_timesteps,
        )

        infectious_weights = self.gnn_model(
            agents_data,
            lam_gamma_integrals,
            self.outbreak_ctl_cfg,
            self.perturbation_flag,
        )

        # --------------------------------------------
        # Step 3: Creating newly infected agents
        # --------------------------------------------
        prob_yes = 1.0 - torch.exp(-infectious_weights)
        potentially_exposed_today = apply_gumbel_softmax(prob_yes, temporal_seed=t)

        newly_exposed_mask = self.progression_model.get_newly_exposed(
            self.current_stages, potentially_exposed_today
        )

        return newly_exposed_mask

    def create_random_exposed_tensors(
        self,
        random_percentage,
    ):
        return self.progression_model.add_random_exposed(
            random_percentage,
            self.current_stages,
        )

    def init_infected_tensors(
        self,
        initial_infected_ids,
        initial_infected_percentage,
        infected_to_recovered_or_dead_time,
        t,
    ):
        proc_initial_infected_ids = None
        if initial_infected_ids is not None:
            proc_initial_infected_ids = initial_infected_ids[t]

        self.current_stages = self.progression_model.init_infected_agents(
            proc_initial_infected_ids,
            initial_infected_percentage,
            self.agents_id,
        )

        self.agents_next_stage_times = (
            self.progression_model.init_agents_next_stage_time(
                self.current_stages, infected_to_recovered_or_dead_time
            ).float()
        )
        self.agents_infected_index = (
            self.current_stages == STAGE_INDEX["infected"]
        ).to(DEVICE)

        self.agents_infected_time = self.progression_model.init_infected_time(
            self.current_stages
        ).float()

    def get_params(self, param_info: dict, param_t: list):
        self.proc_params = {}
        for proc_param in ALL_PARAMS:

            # For prediction, we can set scaler for learnable parameters
            proc_scaler_update = get_learnable_params_scaler(
                proc_param, self.learnable_params_scaler_update
            )

            if proc_param in param_info["learnable_param_order"]:
                try:
                    self.proc_params[proc_param] = (
                        param_t[param_info["learnable_param_order"].index(proc_param)]
                        * proc_scaler_update
                    )
                except IndexError:  # if only one learnable param
                    self.proc_params[proc_param] = param_t
            else:
                self.proc_params[proc_param] = param_info["learnable_param_default"][
                    proc_param
                ]

    def cal_lam_gamma_integrals(
        self,
        shape,
        scale,
        infection_gamma_scaling_factor,
        max_infectiousness_timesteps=3,
    ):

        gamma_dist = torch_gamma(concentration=shape, rate=1 / scale)

        res = (
            gamma_dist.log_prob(
                torch.tensor(range(max([2, int(max_infectiousness_timesteps)])))
            )
            .exp()
            .to(DEVICE)
        )

        if infection_gamma_scaling_factor is not None:
            res_factor = infection_gamma_scaling_factor / max(res.tolist())
            res = res * res_factor
        res[0] = res[1] / 3.0

        self.lam_gamma_integrals = res.to(DEVICE)

    def step(
        self,
        t,
        param_t,
        param_info,
        total_timesteps,
        target,
        save_records: bool = False,
    ):

        random_exposed_mask = None
        if t == 0:
            self.get_params(param_info, param_t)
            self.init_infected_tensors(
                self.initial_infected_ids,
                self.proc_params["initial_infected_percentage"],
                self.proc_params["infected_to_recovered_or_death_time"],
                t,
            )
            self.cal_lam_gamma_integrals(
                self.proc_params["infection_gamma_shape"],
                self.proc_params["infection_gamma_scale"],
                self.proc_params["infection_gamma_scaling_factor"],
            )
        else:
            if self.use_random_exposed:
                random_exposed_mask = self.create_random_exposed_tensors(
                    self.proc_params["random_infected_percentage"],
                )

        newly_exposed_mask = self.get_newly_exposed(
            self.lam_gamma_integrals,
            self.proc_params["vaccine_efficiency_spread"],
            self.proc_params["contact_tracing_coverage"],
            t,
            total_timesteps,
        )

        target, target_indices = self.progression_model.get_target_variables(
            self.proc_params["vaccine_efficiency_symptom"],
            self.current_stages,
            target,
            t,
            total_timesteps,
        )

        if PRINT_MODEL_INFO:
            print(
                f"  {t}: ",
                f"exposed: {self.current_stages.tolist().count(1.0)} |",
                f"infected: {self.current_stages.tolist().count(2.0)} |",
                f"newly exposed: {int(newly_exposed_mask.sum().item())} |",
                # f"infection_ratio_distribution_percentage: {infection_ratio_distribution_percentage}",
                f"target: {int(target[0][t].sum())}",
            )
            print(
                f" - Memory: {round(torch.cuda.memory_allocated(0) / (1024**3), 3) } Gb"
            )

        stage_records = None
        if save_records:
            stage_records = shallow_copy(self.current_stages.tolist())

        next_stages = self.progression_model.update_current_stage(
            newly_exposed_mask,
            random_exposed_mask,
            self.current_stages,
            self.agents_next_stage_times,
            t,
        )

        self.agents_next_stage_times = self.progression_model.update_next_stage_times(
            self.proc_params["exposed_to_infected_time"],
            self.proc_params["infected_to_recovered_or_death_time"],
            newly_exposed_mask,
            self.current_stages,
            self.agents_next_stage_times,
            t,
            total_timesteps,
        )

        self.agents_infected_index = (
            (next_stages == STAGE_INDEX["infected"]).bool().to(DEVICE)
        )

        self.agents_infected_time[newly_exposed_mask.bool()] = (
            t + self.proc_params["exposed_to_infected_time"]
        )

        self.current_stages = next_stages
        self.current_time += 1

        return stage_records, target, target_indices


def build_abm(
    all_agents: DataFrame,
    all_interactions: dict,
    infection_cfg: dict,
    outbreak_ctl_cfg: dict,
    cfg_update: None or dict = None,  # type: ignore
):
    """build simulator: ABM

    Args:
        all_agents (DataFrame): All agents data
        all_interactions (dict): Interaction data in a dict
        infection_cfg (dict): Configuration for infection (under training cfg)
        cfg_update (Noneordict): Configuration to update (e.g., used for prediction)
    """

    def _updated_predict_cfg(
        params: dict, predict_cfg: None or dict  # type: ignore
    ) -> dict:
        """If it is a prediction, update the configuration

        Args:
            params (dict): original parameters
            predict_cfg (Noneordict): configuration related to prediction

        Raises:
            Exception: _description_

        Returns:
            dict: Updated prediction configuration
        """

        if predict_cfg is not None:
            for param_key in [
                "perturbation_flag",
                "use_random_exposed",
                "scaling_factor",
                "outbreak_ctl",
                "initial_infected_ids",
                "learnable_params_scaler",
            ]:
                try:
                    if param_key == "initial_infected_ids":
                        initial_infected_ids_cfg = predict_cfg["initial_infected_ids"]
                        all_infected_ids = {}
                        for proc_t in initial_infected_ids_cfg:
                            proc_value = initial_infected_ids_cfg[proc_t]
                            if isinstance(proc_value, str):
                                if proc_value[-4:] == "json":
                                    with open(proc_value, "r") as json_file:
                                        proc_data = json_load(json_file)

                                    all_infected_ids[proc_t] = list(proc_data["id"])
                                else:
                                    raise Exception(
                                        "Not supported initial infected ID file format"
                                    )
                            elif isinstance(proc_value, list):
                                all_infected_ids[proc_t] = proc_value
                            else:
                                raise Exception(
                                    "Not supported initial infected ID format"
                                )
                        params["predict_update"][
                            f"{param_key}_update"
                        ] = all_infected_ids
                    else:
                        params["predict_update"][f"{param_key}_update"] = predict_cfg[
                            param_key
                        ]
                except (KeyError, TypeError):
                    params["predict_update"][f"{param_key}_update"] = None

        return params

    params = {
        "all_agents": all_agents,
        "all_interactions": all_interactions,
        "infection_cfg": infection_cfg,
        "outbreak_ctl_cfg": outbreak_ctl_cfg,
        "predict_update": {
            "perturbation_flag_update": None,
            "scaling_factor_update": None,
            "outbreak_ctl_cfg_update": None,
            "initial_infected_ids_update": None,
            "use_random_exposed_update": None,
            "learnable_params_scaler_update": None,
        },
    }

    params = _updated_predict_cfg(params, cfg_update)
    abm = GradABM(params)

    return abm


def init_abm(model_inputs: dict, cfg: dict) -> dict:
    """Initiaite an ABM model

    Args:
        model_inputs (dict): ABM model inputs in a dict
        cfg (dict): Training configuration
        prerun_params (listorNone, optional): Pre-run parameters. Defaults to None.

    Returns:
        dict: Initial ABM model
    """
    # logger.info("Building ABM ...")
    abm = build_abm(
        model_inputs["all_agents"],
        model_inputs["all_interactions"],
        cfg["infection"],
        cfg["outbreak_ctl"],
    )

    logger.info("     * Creating initial parameters (to be trained) ...")
    num_epochs = OPTIMIZATION_CFG["num_epochs"]

    param_model = build_param_model(
        cfg["learnable_params"],
        OPTIMIZATION_CFG["use_temporal_params"],
    )

    logger.info("     * Creating loss function ...")
    loss_def = get_loss_func(param_model, model_inputs["total_timesteps"])

    return {
        "num_epochs": num_epochs,
        "param_model": param_model,
        "model": abm,
        "loss_def": loss_def,
    }
