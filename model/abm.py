from abc import ABC, abstractmethod
from copy import copy as shallow_copy
from logging import getLogger

import torch
import torch.nn.functional as F
from numpy import array, isnan, where
from numpy.random import choice
from scipy.stats import gamma as stats_gamma
from torch import manual_seed as torch_seed
from torch import ones_like as torch_ones_like
from torch.distributions import Gamma as torch_gamma
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from model import ALL_PARAMS, DEVICE, STAGE_INDEX, TORCH_SEED_NUM, USE_TEMPORAL_PARAMS
from model.infection_network import InfectionNetwork
from model.seirm_progression import SEIRMProgression
from utils.utils import create_random_seed, round_a_list

logger = getLogger()


# all_agents, all_interactions, num_steps, num_agents
class GradABM:
    def __init__(self, params):
        self.params = params
        self.device = DEVICE

        # Getting agents
        agents_df = self.params["all_agents"]
        # self.agents_id = torch.arange(0, len(agents_df)).long().to(self.device)
        self.agents_id = torch.tensor(agents_df["id"].to_numpy()).long().to(self.device)
        self.agents_ages = torch.tensor(agents_df["age"].to_numpy()).long().to(self.device)
        self.agents_sex = torch.tensor(agents_df["sex"].to_numpy()).long().to(self.device)
        self.agents_ethnicity = (
            torch.tensor(agents_df["ethnicity"].to_numpy()).long().to(self.device)
        )
        self.agents_vaccine = torch.tensor(agents_df["vaccine"].to_numpy()).long().to(self.device)
        self.agents_area = torch.tensor(agents_df["area"].to_numpy()).long().to(self.device)
        self.scaling_factor_update = self.params["scaling_factor_update"]
        self.outbreak_ctl_update = self.params["outbreak_ctl_update"]
        self.initial_infected_sa2 = self.params["initial_infected_sa2"]

        self.params["num_agents"] = len(agents_df)
        self.num_agents = self.params["num_agents"]
        self.use_random_infection = True

        if self.params["use_random_infection"] is not None:
            self.use_random_infection = self.params["use_random_infection"]

        # Getting interaction:
        all_interactions = self.params["all_interactions"]
        self.agents_mean_interactions_mu = all_interactions["agents_mean_interactions_mu"]
        self.agents_mean_interactions_mu_split = all_interactions[
            "agents_mean_interactions_mu_split"
        ]
        self.network_type_dict_inv = all_interactions["network_type_dict_inv"]
        self.network_type_dict = all_interactions["network_type_dict"]

        self.DPM = SEIRMProgression(self.params)

        self.SFSusceptibility_age = (
            torch.tensor(self.params["infection_cfg"]["scaling_factor"]["age_dependant"])
            # torch.tensor([0.1])
            .float().to(self.device)
        )
        self.SFInfector = (
            torch.tensor(self.params["infection_cfg"]["scaling_factor"]["symptom_dependant"])
            .float()
            .to(self.device)
        )

        self.SFSusceptibility_ethnicity = (
            torch.tensor(self.params["infection_cfg"]["scaling_factor"]["ethnicity_dependant"])
            .float()
            .to(self.device)
        )

        self.SFSusceptibility_sex = (
            torch.tensor(self.params["infection_cfg"]["scaling_factor"]["sex_dependant"])
            .float()
            .to(self.device)
        )

        # Scaling factor for vaccine
        try:
            vaccine_cfg = self.scaling_factor_update["vaccine"]
        except (KeyError, TypeError):
            vaccine_cfg = self.params["infection_cfg"]["scaling_factor"]["vaccine"]

        self.SFSusceptibility_vaccine = torch.tensor(vaccine_cfg).float().to(self.device)

        # Scaling factor for outbreak ctl
        try:
            self.outbreak_ctl_cfg = self.outbreak_ctl_update["outbreak_ctl"]
        except (KeyError, TypeError):
            self.outbreak_ctl_cfg = self.params["infection_cfg"]["outbreak_ctl"]

        # Scaling factor perturbation for outbreak ctl
        try:
            self.perturbation = self.outbreak_ctl_update["perturbation"]
        except (KeyError, TypeError):
            self.perturbation = False

        self.net = InfectionNetwork(
            self.SFSusceptibility_age,
            self.SFSusceptibility_sex,
            self.SFSusceptibility_ethnicity,
            self.SFSusceptibility_vaccine,
            self.SFInfector,
            # self.lam_gamma_integrals,
            self.device,
        ).to(self.device)

        self.current_time = 0
        self.all_edgelist = all_interactions["all_edgelist"]
        self.all_edgeattr = all_interactions["all_edgeattr"]

    def get_newly_exposed(self, lam_gamma_integrals, t):
        all_nodeattr = torch.stack(
            (
                self.agents_ages,  # 0: age
                self.agents_sex,  # 1: sex
                self.agents_ethnicity,  # 2: ethnicity
                self.agents_vaccine,  # 3: vaccine
                self.current_stages.detach(),  # 4: stage
                self.agents_infected_index.to(self.device),  # 5: infected index
                self.agents_infected_time.to(self.device),  # 6: infected time
                *self.agents_mean_interactions_mu_split,  # 7 to 29: represents the number of venues where agents can interact with each other
                torch.arange(self.params["num_agents"]).to(self.device),  # Agent ids (30)
            )
        ).t()

        agents_data = Data(
            all_nodeattr,
            edge_index=self.all_edgelist,
            edge_attr=self.all_edgeattr,
            t=t,
            agents_mean_interactions=self.agents_mean_interactions_mu,
        )

        # print(t, f"before: {round(torch.cuda.memory_allocated(0) / (1024**3), 3) } Gb")
        lam_t = self.net(
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

        while True:
            if TORCH_SEED_NUM is not None:
                # torch_seed(create_random_seed())
                torch_seed(TORCH_SEED_NUM["newly_exposed"])

            potentially_exposed_today = F.gumbel_softmax(
                logits=cat_logits, tau=1, hard=True, dim=1
            )[:, 0]

            if not isnan(potentially_exposed_today.cpu().clone().detach().numpy()).any():
                break

        newly_exposed_today = self.DPM.get_newly_exposed(
            self.current_stages, potentially_exposed_today
        )

        return newly_exposed_today

    def create_random_infected_tensors(
        self,
        random_percentage,
        infected_to_recovered_time,
        t,
    ):
        return self.DPM.add_random_infected(
            random_percentage,
            infected_to_recovered_time,
            self.current_stages,
            self.agents_infected_time,
            self.agents_next_stage_times,
            t,
            self.device,
        )

    def init_infected_tensors(
        self, initial_infected_percentage, infected_to_recovered_or_dead_time, agents_area
    ):
        self.current_stages = self.DPM.init_infected_agents(
            initial_infected_percentage,
            agents_area,
            self.initial_infected_sa2,
            self.device,
        )

        self.agents_next_stage_times = self.DPM.init_agents_next_stage_time(
            self.current_stages,
            infected_to_recovered_or_dead_time,
            self.device,
        ).float()

        self.agents_infected_index = (self.current_stages == STAGE_INDEX["infected"]).to(
            self.device
        )

        self.agents_infected_time = self.DPM.init_infected_time(
            self.current_stages, self.device
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
            recovered_or_death_num = current_stages_list.count(STAGE_INDEX["recovered_or_death"])
            total_num = susceptible_num + exposed_num + infected_num + recovered_or_death_num
            print(
                f"    cur_stage: {susceptible_num}(susceptible), {exposed_num}(exposed), {infected_num}(infected), {recovered_or_death_num}(recovered_or_death), {total_num}(total)"
            )
        if "stage_times" in debug_info:
            print(f"    next_stage_times: {round_a_list(agents_next_stage_times.tolist())}")
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
                self.proc_params[proc_param] = param_info["learnable_param_default"][proc_param]

    def cal_lam_gamma_integrals(
        self, shape, scale, infection_gamma_scaling_factor, max_infectiousness_timesteps=3
    ):
        self.lam_gamma_integrals = self._get_lam_gamma_integrals(
            shape,
            scale,
            infection_gamma_scaling_factor,
            int(max_infectiousness_timesteps),
        ).to(self.device)

    def step(
        self,
        t,
        param_t,
        param_info,
        total_timesteps,
        debug: bool = False,
        save_records: bool = False,
    ):
        if t == 0:
            self.get_params(param_info, param_t)
            self.init_infected_tensors(
                self.proc_params["initial_infected_percentage"],
                self.proc_params["infected_to_recovered_or_death_time"],
                self.agents_area,
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

        newly_exposed_today = self.get_newly_exposed(self.lam_gamma_integrals, t)

        recovered_dead_now, death_indices, target = self.DPM.get_target_variables(
            self.proc_params["vaccine_efficiency"],
            self.current_stages,
            self.agents_next_stage_times,
            self.proc_params["infected_to_recovered_or_death_time"],
            t,
        )

        """
        if 165400 in self.agents_area[death_indices]:
            import numpy

            agent_loc = numpy.where(
                numpy.array(self.agents_area[death_indices].tolist()) == 165400
            )[0][0]
            print(
                self.agents_ethnicity[death_indices[agent_loc]],
                self.agents_ages[death_indices[agent_loc]],
            )

        print(
            t,
            "165400: ",
            165400 in self.agents_area[death_indices],
            "147200: ",
            147200 in self.agents_area[death_indices],
        )
        """
        """
        print(
            t,
            f"exposed: {self.current_stages.tolist().count(1.0)}",
            f"infected: {self.current_stages.tolist().count(2.0)}",
            f"newly exposed: {int(newly_exposed_today.sum().item())}",
            f"death: {int(target)}",
        )
        """
        # print(t, target)
        stage_records = None
        if save_records:
            stage_records = shallow_copy(self.current_stages.tolist())

        if debug:
            self.print_debug_info(
                self.current_stages, self.agents_next_stage_times, newly_exposed_today, target, t
            )

        # get next stages without updating yet the current_stages
        next_stages = self.DPM.update_current_stage(
            newly_exposed_today,
            self.current_stages,
            self.agents_next_stage_times,
            death_indices,
            t,
        )

        # update times with current_stages
        self.agents_next_stage_times = self.DPM.update_next_stage_times(
            self.proc_params["exposed_to_infected_time"],
            self.proc_params["infected_to_recovered_or_death_time"],
            newly_exposed_today,
            self.current_stages,
            self.agents_next_stage_times,
            t,
            total_timesteps,
        )

        self.agents_infected_index = (
            (next_stages == STAGE_INDEX["infected"]).bool().to(self.device)
        )

        # self.agents_infected_index = (
        #    (self.current_stages == STAGE_INDEX["infected"]).bool().to(self.device)
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

        res = gamma_dist.log_prob(torch.tensor(range(total_t))).exp().to(self.device)

        if infection_gamma_scaling_factor is not None:
            res_factor = infection_gamma_scaling_factor / max(res.tolist())
            res = res * res_factor
        res[0] = res[1] / 3.0
        return res


def forward_abm(param_values_all, param_info, abm, training_num_steps, save_records: bool = False):
    predictions = []
    all_records = []
    all_target_indices = []

    param_values_all = param_values_all.to(abm.device)

    for time_step in range(training_num_steps):
        if USE_TEMPORAL_PARAMS:
            param_values = param_values_all[0, time_step, :].to(abm.device)
        else:
            param_values = param_values_all

        proc_record, target_indices, pred_t = abm.step(
            time_step,
            param_values,
            param_info,
            training_num_steps,
            debug=False,
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
        all_records = array(all_records)

    return {
        "prediction": predictions,
        "all_records": all_records,
        "all_target_indices": all_target_indices,
        "agents_area": abm.agents_area.tolist(),
        "agents_ethnicity": abm.agents_ethnicity.tolist(),
    }


def build_abm(all_agents, all_interactions, infection_cfg: dict, cfg_update: None or dict):
    """build simulator: ABM or ODE"""
    params = {
        "all_agents": all_agents,
        "all_interactions": all_interactions,
        "infection_cfg": infection_cfg,
        "scaling_factor_update": None,
        "outbreak_ctl_update": None,
        "initial_infected_sa2": None,
        "use_random_infection": None,
    }

    if cfg_update is not None:
        for param_key in [
            "use_random_infection",
            "scaling_factor_update",
            "outbreak_ctl_update",
            "initial_infected_sa2",
        ]:
            try:
                params[param_key] = cfg_update[param_key]
            except KeyError:
                params[param_key] = None
    abm = GradABM(params)

    return abm
