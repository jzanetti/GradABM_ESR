from abc import ABC, abstractmethod
from collections import Counter
from copy import copy as shallow_copy
from logging import getLogger

import torch
import torch.nn.functional as F
from numpy import array, isnan, where
from numpy.random import choice
from pandas import DataFrame
from pandas import read_csv as pandas_read_csv
from scipy.stats import gamma as stats_gamma
from torch import manual_seed as torch_seed
from torch import ones_like as torch_ones_like
from torch.distributions import Gamma as torch_gamma
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from process.model import (
    ALL_PARAMS,
    DEVICE,
    INITIAL_LOSS,
    OPTIMIZATION_CFG,
    PRERUN_NUM_EPOCHS,
    PRINT_INCRE,
    STAGE_INDEX,
    TORCH_SEED_NUM,
)
from process.model.gnn import GNN_model
from process.model.loss_func import get_loss_func, loss_optimization
from process.model.param import build_param_model, obtain_param_cfg, param_model_forward
from process.model.prep import update_params_for_prerun
from process.model.progression import Progression_model
from utils.utils import create_random_seed, round_a_list

logger = getLogger()


class GradABM:
    def __init__(self, params):
        # -----------------------------
        # Step 2: set up agents
        # -----------------------------
        for attr in ["id", "age", "gender", "ethnicity", "vaccine", "area"]:
            setattr(
                self,
                f"agents_{attr}",
                torch.tensor(params["all_agents"][attr].to_numpy()).long().to(DEVICE),
            )
        self.num_agents = len(params["all_agents"])

        # -----------------------------
        # Step 3: set up interaction
        # -----------------------------
        all_interactions = params["all_interactions"]
        self.agents_mean_interactions_mu = all_interactions[
            "agents_mean_interactions_mu"
        ]
        self.agents_mean_interactions_mu_split = all_interactions[
            "agents_mean_interactions_mu_split"
        ]
        self.all_edgelist = all_interactions["all_edgelist"]
        self.all_edgeattr = all_interactions["all_edgeattr"]

        # -----------------------------
        # Step 4: set up prediction configuration
        # -----------------------------
        self.initial_infected_ids = params["initial_infected_ids"]
        self.use_random_infection = True
        if params["use_random_infection"] is not None:
            self.use_random_infection = params["use_random_infection"]

        self.outbreak_ctl_update = params["outbreak_ctl_update"]
        try:
            self.outbreak_ctl_cfg = self.outbreak_ctl_update["outbreak_ctl"]
        except (KeyError, TypeError):
            self.outbreak_ctl_cfg = params["infection_cfg"]["outbreak_ctl"]

        try:
            self.perturbation = self.outbreak_ctl_update["perturbation"]
        except (KeyError, TypeError):
            self.perturbation = False

        # -----------------------------
        # Step 5: set up scaling_factor
        # -----------------------------
        for attr in ["age", "symptom", "ethnicity", "gender", "vaccine"]:
            if attr == "vaccine":
                try:
                    vaccine_cfg = params["scaling_factor_update"]["vaccine"]
                except (KeyError, TypeError):
                    vaccine_cfg = params["infection_cfg"]["scaling_factor"]["vaccine"]
                params["infection_cfg"]["scaling_factor"][
                    f"{attr}_dependant"
                ] = vaccine_cfg

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
        # Step 7: set up progress and GNN model
        # -----------------------------
        self.progression_wrapper = Progression_model(self.num_agents)

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
    ):
        all_nodeattr = torch.stack(
            (
                self.agents_age,  # 0: age
                self.agents_gender,  # 1: sex
                self.agents_ethnicity,  # 2: ethnicity
                self.agents_vaccine,  # 3: vaccine
                self.current_stages.detach(),  # 4: stage
                self.agents_infected_index.to(DEVICE),  # 5: infected index
                self.agents_infected_time.to(DEVICE),  # 6: infected time
                *self.agents_mean_interactions_mu_split,  # 7 to 29: represents the number of venues where agents can interact with each other
                torch.arange(self.num_agents).to(DEVICE),  # Agent ids (30)
            )
        ).t()

        agents_data = Data(
            all_nodeattr,
            edge_index=self.all_edgelist,
            edge_attr=self.all_edgeattr,
            vaccine_efficiency_spread=vaccine_efficiency_spread,
            contact_tracing_coverage=contact_tracing_coverage,
            t=t,
            agents_mean_interactions=self.agents_mean_interactions_mu,
        )

        # print(t, f"before: {round(torch.cuda.memory_allocated(0) / (1024**3), 3) } Gb")
        # self.gnn_model goes to forward() in gnn_model function
        lam_t = self.gnn_model(
            agents_data,
            lam_gamma_integrals,
            self.outbreak_ctl_cfg,
            self.perturbation,
        )
        # lam_t = lam_t.to("cpu")
        # print(t, f"after: {round(torch.cuda.memory_allocated(0) / (1024**3), 3) } Gb")

        prob_not_infected = torch.exp(-lam_t)
        p = torch.hstack((1 - prob_not_infected, prob_not_infected))
        cat_logits = torch.log(p + 1e-9)

        infection_distribution_percentage = None
        # infection_distribution_percentage = {
        #    value: count / len(lam_t[:, 0].tolist()) * 100
        #    for value, count in Counter(lam_t[:, 0].tolist()).items()
        # }
        while True:
            if TORCH_SEED_NUM is not None:
                # torch_seed(create_random_seed())
                torch_seed(TORCH_SEED_NUM["newly_exposed"])

            potentially_exposed_today = F.gumbel_softmax(
                logits=cat_logits, tau=1, hard=True, dim=1
            )[:, 0]

            if not isnan(
                potentially_exposed_today.cpu().clone().detach().numpy()
            ).any():
                break

        newly_exposed_today = self.progression_wrapper.get_newly_exposed(
            self.current_stages, potentially_exposed_today
        )
        # newly_exposed_today = potentially_exposed_today
        return (
            newly_exposed_today,
            potentially_exposed_today,
            infection_distribution_percentage,
        )

    def create_random_infected_tensors(
        self,
        random_percentage,
        infected_to_recovered_time,
        t,
    ):
        return self.progression_wrapper.add_random_infected(
            random_percentage,
            infected_to_recovered_time,
            self.current_stages,
            self.agents_infected_time,
            self.agents_next_stage_times,
            t,
            # self.device,
        )

    def init_infected_tensors(
        self,
        initial_infected_percentage,
        infected_to_recovered_or_dead_time,
    ):
        self.current_stages = self.progression_wrapper.init_infected_agents(
            initial_infected_percentage,
            self.initial_infected_ids,
            self.agents_id,
            # self.device,
        )
        self.agents_next_stage_times = self.progression_wrapper.init_agents_next_stage_time(
            self.current_stages,
            infected_to_recovered_or_dead_time,
            # self.device,
        ).float()

        self.agents_infected_index = (
            self.current_stages == STAGE_INDEX["infected"]
        ).to(DEVICE)

        self.agents_infected_time = self.progression_wrapper.init_infected_time(
            self.current_stages,  # self.device
        ).float()

    def print_debug_info(
        self,
        current_stages,
        agents_next_stage_times,
        newly_exposed_today,
        target2,
        t,
        debug_info: list = ["t", "stage", "die"],
    ):
        if "t" in debug_info:
            print(f"Timestep: {t}")
        if "stage" in debug_info:
            current_stages_list = current_stages.tolist()
            susceptible_num = current_stages_list.count(STAGE_INDEX["susceptible"])
            exposed_num = current_stages_list.count(STAGE_INDEX["exposed"])
            infected_num = current_stages_list.count(STAGE_INDEX["infected"])
            recovered_or_death_num = current_stages_list.count(
                STAGE_INDEX["recovered_or_death"]
            )
            total_num = (
                susceptible_num + exposed_num + infected_num + recovered_or_death_num
            )
            print(
                f"    cur_stage: {susceptible_num}(susceptible), {exposed_num}(exposed), {infected_num}(infected), {recovered_or_death_num}(recovered_or_death), {total_num}(total)"
            )
        if "stage_times" in debug_info:
            print(
                f"    next_stage_times: {round_a_list(agents_next_stage_times.tolist())}"
            )
        if "exposed" in debug_info:
            print(f"    newly exposed: {newly_exposed_today}")
        if "die" in debug_info:
            print(f"    die: {target2}")

    def get_params(self, param_info: dict, param_t: list):
        self.proc_params = {}
        for proc_param in ALL_PARAMS:
            if proc_param in param_info["learnable_param_order"]:
                try:
                    self.proc_params[proc_param] = param_t[
                        param_info["learnable_param_order"].index(proc_param)
                    ]
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
        self.lam_gamma_integrals = self._get_lam_gamma_integrals(
            shape,
            scale,
            infection_gamma_scaling_factor,
            int(max_infectiousness_timesteps),
        ).to(DEVICE)

    def step(
        self,
        t,
        param_t,
        param_info,
        total_timesteps,
        debug: bool = False,
        save_records: bool = False,
        print_step_info: bool = False,
    ):
        if t == 0:
            self.get_params(param_info, param_t)
            self.init_infected_tensors(
                self.proc_params["initial_infected_percentage"],
                self.proc_params["infected_to_recovered_or_death_time"],
            )
            self.cal_lam_gamma_integrals(
                self.proc_params["infection_gamma_shape"],
                self.proc_params["infection_gamma_scale"],
                self.proc_params["infection_gamma_scaling_factor"],
            )
        else:
            if self.use_random_infection:
                (
                    self.current_stages,
                    self.agents_infected_time,
                    self.agents_next_stage_times,
                ) = self.create_random_infected_tensors(
                    self.proc_params["random_infected_percentgae"],
                    self.proc_params["infected_to_recovered_or_death_time"],
                    t,
                )

        (
            newly_exposed_today,
            potentially_exposed_today,
            infection_ratio_distribution_percentage,
        ) = self.get_newly_exposed(
            self.lam_gamma_integrals,
            self.proc_params["vaccine_efficiency_spread"],
            self.proc_params["contact_tracing_coverage"],
            t,
        )

        (
            potential_infected,
            death_indices,
            target,
        ) = self.progression_wrapper.get_target_variables(
            self.proc_params["vaccine_efficiency_symptom"],
            self.current_stages,
            self.agents_next_stage_times,
            self.proc_params["infected_to_recovered_or_death_time"],
            t,
        )

        if print_step_info:
            print(
                t,
                f"exposed: {self.current_stages.tolist().count(1.0)} |",
                f"infected: {self.current_stages.tolist().count(2.0)} |",
                f"newly exposed: {int(newly_exposed_today.sum().item())} |",
                f"potential newly exposed: {int(potentially_exposed_today.sum().item())} |",
                # f"infection_ratio_distribution_percentage: {infection_ratio_distribution_percentage}",
                f"target: {int(target)}",
            )
            # print(f" - Memory: {round(torch.cuda.memory_allocated(0) / (1024**3), 3) } Gb")

        stage_records = None
        if save_records:
            stage_records = shallow_copy(self.current_stages.tolist())

        if debug:
            self.print_debug_info(
                self.current_stages,
                self.agents_next_stage_times,
                newly_exposed_today,
                target,
                t,
            )

        next_stages = self.progression_wrapper.update_current_stage(
            newly_exposed_today,
            self.current_stages,
            self.agents_next_stage_times,
            death_indices,
            t,
        )
        # update times with current_stages
        self.agents_next_stage_times = self.progression_wrapper.update_next_stage_times(
            self.proc_params["exposed_to_infected_time"],
            self.proc_params["infected_to_recovered_or_death_time"],
            newly_exposed_today,
            self.current_stages,
            self.agents_next_stage_times,
            t,
            total_timesteps,
        )

        self.agents_infected_index = (
            (next_stages == STAGE_INDEX["infected"]).bool().to(DEVICE)
        )

        # self.agents_infected_index = (
        #    (self.current_stages == STAGE_INDEX["infected"]).bool().to(DEVICE)
        # )
        # self.agents_infected_index[recovered_dead_now.bool()] = False
        self.agents_infected_time[newly_exposed_today.bool()] = (
            t + self.proc_params["exposed_to_infected_time"]
        )

        self.current_stages = next_stages
        self.current_time += 1

        return stage_records, death_indices, target

    def _get_lam_gamma_integrals(self, a, b, infection_gamma_scaling_factor, total_t):
        total_t = max([2, total_t])

        gamma_dist = torch_gamma(concentration=a, rate=1 / b)

        res = gamma_dist.log_prob(torch.tensor(range(total_t))).exp().to(DEVICE)

        if infection_gamma_scaling_factor is not None:
            res_factor = infection_gamma_scaling_factor / max(res.tolist())
            res = res * res_factor
        res[0] = res[1] / 3.0
        return res


def build_abm(
    all_agents: DataFrame,
    all_interactions: dict,
    infection_cfg: dict,
    cfg_update: None or dict = None,
):
    """build simulator: ABM

    Args:
        all_agents (DataFrame): All agents data
        all_interactions (dict): Interaction data in a dict
        infection_cfg (dict): Configuration for infection (under training cfg)
        cfg_update (Noneordict): Configuration to update (e.g., used for prediction)
    """

    def _updated_predict_cfg(params: dict, predict_cfg: None or dict) -> dict:
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
                "use_random_infection",
                "scaling_factor_update",
                "outbreak_ctl_update",
                "initial_infected_ids",
            ]:
                try:
                    if param_key == "initial_infected_ids":
                        initial_infected_ids_cfg = predict_cfg["initial_infected_ids"]
                        all_infected_ids = []
                        for proc_agent in initial_infected_ids_cfg:
                            if isinstance(proc_agent, str):
                                if proc_agent.endswith("csv"):
                                    proc_data = pandas_read_csv(proc_agent)
                                    all_infected_ids.extend(list(proc_data["id"]))
                                else:
                                    raise Exception(
                                        "Not able to get the cfg for initial_infected_ids"
                                    )
                            else:
                                all_infected_ids.append(proc_agent)
                        params[param_key] = all_infected_ids
                    else:
                        params[param_key] = predict_cfg[param_key]
                except (KeyError, TypeError):
                    params[param_key] = None

        return params

    params = {
        "all_agents": all_agents,
        "all_interactions": all_interactions,
        "infection_cfg": infection_cfg,
        "scaling_factor_update": None,
        "outbreak_ctl_update": None,
        "initial_infected_ids": None,
        "use_random_infection": None,
    }

    params = _updated_predict_cfg(params, cfg_update)
    abm = GradABM(params)

    return abm


def init_abm(
    model_inputs: dict,
    cfg: dict,
    prerun_params: list or None = None,
) -> dict:
    """Initiaite an ABM model

    Args:
        model_inputs (dict): ABM model inputs in a dict
        cfg (dict): Training configuration
        prerun_params (listorNone, optional): Pre-run parameters. Defaults to None.

    Returns:
        dict: Initial ABM model
    """
    logger.info("Building ABM ...")
    abm = build_abm(
        model_inputs["all_agents"], model_inputs["all_interactions"], cfg["infection"]
    )

    logger.info("Creating initial parameters (to be trained) ...")
    param_model = build_param_model(
        obtain_param_cfg(cfg["learnable_params"], prerun_params),
        OPTIMIZATION_CFG["use_temporal_params"],
    )

    logger.info("Creating loss function ...")
    loss_def = get_loss_func(param_model, model_inputs["total_timesteps"])

    if prerun_params:
        num_epochs = PRERUN_NUM_EPOCHS
        cfg = update_params_for_prerun(cfg)
    else:
        num_epochs = OPTIMIZATION_CFG["num_epochs"]

    return {
        "num_epochs": num_epochs,
        "param_model": param_model,
        "model": abm,
        "loss_def": loss_def,
    }
