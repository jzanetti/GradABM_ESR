import os
from abc import ABC, abstractmethod
from copy import copy

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.stats import gamma
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from model.utils import get_dir_from_path_list


class DiseaseProgression(ABC):
    """abstract class"""

    def __init__(self):
        pass

    @abstractmethod
    def init_infected_agents(self):
        """initialize tensor variables depending on disease"""
        pass

    @abstractmethod
    def update_next_stage_times(self):
        """update time"""
        pass

    @abstractmethod
    def update_current_stage(self):
        """update stage"""
        pass


class SEIRMProgression(DiseaseProgression):
    """SEIRM for COVID-19"""

    def __init__(self, params):
        super(DiseaseProgression, self).__init__()
        # encoding of stages
        # Stage progress:
        # SUSCEPTIBLE => EXPOSED => INFECTED => RECOVERED/MORTALITY
        self.SUSCEPTIBLE_VAR = 0
        self.EXPOSED_VAR = 1
        self.INFECTED_VAR = 2
        self.RECOVERED_VAR = 3
        self.MORTALITY_VAR = 4
        # default times (only for initialization, later they are learned)
        self.EXPOSED_TO_INFECTED_TIME = 3
        self.INFECTED_TO_RECOVERED_TIME = 5
        # inf time
        # self.INFINITY_TIME = params["num_steps"] + 1
        self.num_agents = params["num_agents"]

    def init_infected_agents(self, initial_infected_percentage, device):
        prob_infected = (initial_infected_percentage * 0.01) * torch.ones((self.num_agents, 1)).to(
            device
        )
        p = torch.hstack((prob_infected, 1 - prob_infected))
        cat_logits = torch.log(p + 1e-9)
        agents_stages = F.gumbel_softmax(logits=cat_logits, tau=1, hard=True, dim=1)[:, 0]
        agents_stages *= self.INFECTED_VAR
        agents_stages = agents_stages.to(device)
        return agents_stages

    def init_infected_time(self, agents_stages, device):
        agents_infected_time = -999 * torch.ones_like(agents_stages).to(device)
        agents_infected_time[agents_stages == self.INFECTED_VAR] = 0

        return agents_infected_time

    def init_agents_next_stage_time(
        self, agents_stages, infected_to_recovered_or_dead_time, device
    ):
        agents_next_stage_times = 0.001 * torch.ones_like(agents_stages).long().to(device)
        agents_next_stage_times[agents_stages == self.INFECTED_VAR] = (
            0 + infected_to_recovered_or_dead_time
        )
        return agents_next_stage_times

    def update_initial_times(self, learnable_params, agents_stages, agents_next_stage_times):
        """this is for the abm constructor"""
        infected_to_recovered_time = learnable_params["infected_to_recovered_time"]
        exposed_to_infected_time = learnable_params["exposed_to_infected_time"]
        agents_next_stage_times[agents_stages == self.EXPOSED_VAR] = exposed_to_infected_time
        agents_next_stage_times[agents_stages == self.INFECTED_VAR] = infected_to_recovered_time
        return agents_next_stage_times

    def get_newly_exposed(self, current_stages, potentially_exposed_today):
        # we now get the ones that new to exposure
        newly_exposed_today = (current_stages == self.SUSCEPTIBLE_VAR) * potentially_exposed_today
        return newly_exposed_today

    def update_next_stage_times(
        self, learnable_params, newly_exposed_today, current_stages, agents_next_stage_times, t
    ):
        """update time

        exposed_to_infected_time = learnable_params["exposed_to_infected_time"]
        infected_to_recovered_time = learnable_params["infected_to_recovered_or_dead_time"]

        new_transition_times = torch.clone(agents_next_stage_times)
        curr_stages = torch.clone(current_stages).long()

        for i in range(len(new_transition_times)):
            if curr_stages[i] == self.INFECTED_VAR and agents_next_stage_times[i] == t:
                new_transition_times[i] = 1  # self.INFINITY_TIME
            elif curr_stages[i] == self.EXPOSED_VAR and agents_next_stage_times[i] == t:
                new_transition_times[i] = t + infected_to_recovered_time

        result = newly_exposed_today * (t + 1 + exposed_to_infected_time) + (1 - newly_exposed_today) * new_transition_times
        return result

        """
        exposed_to_infected_time = learnable_params["exposed_to_infected_time"]
        infected_to_recovered_time = learnable_params["infected_to_recovered_or_dead_time"]
        # for non-exposed
        # if S, R, M -> set to default value; if E/I -> update time if your transition time arrived in the current time
        new_transition_times = torch.clone(agents_next_stage_times)
        curr_stages = torch.clone(current_stages).long()
        new_transition_times[
            (curr_stages == self.INFECTED_VAR) * (agents_next_stage_times <= t)
        ] = 1  # self.INFINITY_TIME
        new_transition_times[
            (curr_stages == self.EXPOSED_VAR) * (agents_next_stage_times <= t)
        ] = (t + infected_to_recovered_time)
        return (
            newly_exposed_today * (t + 1 + exposed_to_infected_time)
            + (1 - newly_exposed_today) * new_transition_times
        )

    def get_target_variables(
        self, learnable_params, current_stages, agents_next_stage_times, t, device
    ):
        """get recovered (not longer infectious) + targets"""

        # recovered_or_dead_today = []
        # for i in range(len(current_stages)):
        #    if current_stages[i] == self.INFECTED_VAR and agents_next_stage_times[i] == t:
        #        recovered_or_dead_today.append(True)
        #    else:
        #        recovered_or_dead_today.append(False)

        # recovered_or_dead_today = []
        # for i in range(len(current_stages)):
        #     if current_stages[i] == self.INFECTED_VAR and agents_next_stage_times[i] == t:
        #        recovered_or_dead_today.append(True)
        #    else:
        #        recovered_or_dead_today.append(False)
        #
        # convert True/False to 1.0/0.0
        # recovered_or_dead_today = torch.tensor(
        #    [float(int(value)) for value in recovered_or_dead_today], requires_grad=True
        # ).to(device)

        mortality_rate = learnable_params["mortality_rate"] / 100.0

        recovered_or_dead_today = F.sigmoid(t - agents_next_stage_times)
        # recovered_or_dead_today = (
        #    recovered_or_dead_today - torch.min(recovered_or_dead_today)
        # ) / (torch.max(recovered_or_dead_today) - torch.min(recovered_or_dead_today))
        # recovered_or_dead_today = current_stages * recovered_or_dead_today

        # print(recovered_or_dead_today.sum())

        # Define the value of INFECTED_VAR
        INFECTED_VAR = torch.tensor(self.INFECTED_VAR)

        # Create the condition mask for current_stages
        # torch.eq will break the computation graph and the output will not have a valid grad_fn associated with it as seen here (see: https://discuss.pytorch.org/t/custom-loss-function-error-element-0-of-tensors-does-not-require-grad-and-does-not-have-grad-fn/87944/9)
        # condition_mask = torch.tensor(
        #     torch.eq(current_stages, INFECTED_VAR).float(), requires_grad=True
        # )
        # condition_mask.requires_grad = True

        # Apply the threshold condition on agents_next_stage_times
        # threshold_mask = (agents_next_stage_times <= t).float()

        # Compute the result with differentiable operations
        # recovered_or_dead_today = (current_stages * condition_mask * threshold_mask) / INFECTED_VAR
        #  recovered_or_dead_today = condition_mask

        # recovered_or_dead_today_old = (
        #    current_stages * (current_stages == self.INFECTED_VAR) * (agents_next_stage_times <= t)
        # ) / self.INFECTED_VAR  # agents when stage changes

        # if recovered_or_dead_today_old.tolist() != recovered_or_dead_today.tolist():
        #    raise Exception("Errorrrrrrr !")

        import numpy as np

        def _continuous_approximation(x):
            return 1 / (1 + torch.exp(-10 * (x + 1)))
            # return (1 - torch.exp(-x)) / (1 + torch.exp(-x + 10))

        if_reached_next_stage = _continuous_approximation(t - agents_next_stage_times)
        # x = torch.where(agents_next_stage_times <= t, torch.tensor(1), torch.tensor(0))
        if_infected = torch.where(
            current_stages == self.INFECTED_VAR, torch.tensor(1), current_stages
        )
        recovered_or_dead_today = torch.mul(if_reached_next_stage, if_infected)
        # recovered_or_dead_today = current_stages * (current_stages == self.INFECTED_VAR)

        death_total_today = mortality_rate * torch.sum(recovered_or_dead_today)
        # death_total_today = recovered_or_dead_today.sum()
        # death_total_today = mortality_rate * recovered_or_dead_today.sum()
        # death_total_today = mortality_rate * newly_exposed_today.sum()

        return recovered_or_dead_today, death_total_today

    def update_current_stage(
        self, newly_exposed_today, current_stages, agents_next_stage_times, t
    ):
        """progress disease: move agents to different disease stage"""

        # transition_from_exposed = []
        # for i in len(all_agents):
        #    proc_next_time = agents_next_stage_times[i]
        #    if t < proc_next_time:
        #        transition_to_from_infected.append(self.EXPOSED_VAR)
        #    else:
        #        transition_to_from_infected.append(self.INFECTED_VAR)

        # Apply differentiable approximations using the sigmoid function
        # after_exposed = self.EXPOSED_VAR * (agents_next_stage_times > t) + self.INFECTED_VAR * (
        #    agents_next_stage_times <= t
        # )

        condition1 = torch.round(torch.sigmoid(10 * (agents_next_stage_times - t)) - 0.001)
        condition2 = torch.round(torch.sigmoid(10 * (t - agents_next_stage_times)) + 0.001)
        after_exposed = self.EXPOSED_VAR * condition1 + self.INFECTED_VAR * condition2

        # after_infected = []
        # for i in len(all_agents):
        #    proc_next_time = agents_next_stage_times[i]
        #    if t < proc_next_time:
        #        after_infected.append(self.INFECTED_VAR)
        #    else:
        #        after_infected.append(self.RECOVERED_VAR)

        # after_infected = self.INFECTED_VAR * (agents_next_stage_times > t) + self.RECOVERED_VAR * (
        #    agents_next_stage_times <= t
        # )
        # condition1 = torch.round(torch.sigmoid(10 * (agents_next_stage_times - t)) - 0.001)
        # condition2 = torch.round(touch.sigmoid(10 * (t - agents_next_stage_times)) + 0.001)
        after_infected = self.INFECTED_VAR * condition1 + self.RECOVERED_VAR * condition2

        # stage_progression = []
        # for stage in current_stages:
        #    if stage == self.SUSCEPTIBLE_VAR:
        #        stage_progression.append(self.SUSCEPTIBLE_VAR)
        #    elif stage == self.RECOVERED_VAR:
        #        stage_progression.append(self.RECOVERED_VAR)
        #    elif stage == self.MORTALITY_VAR:
        #        stage_progression.append(self.MORTALITY_VAR)
        #    elif stage == self.EXPOSED_VAR:
        #        stage_progression.append(transition_to_infected)
        #    elif stage == self.INFECTED_VAR:
        #        stage_progression.append(transition_to_mortality_or_recovered)

        stage_progression = (
            (current_stages == self.SUSCEPTIBLE_VAR) * self.SUSCEPTIBLE_VAR
            + (current_stages == self.RECOVERED_VAR) * self.RECOVERED_VAR
            + (current_stages == self.MORTALITY_VAR) * self.MORTALITY_VAR
            + (current_stages == self.EXPOSED_VAR) * after_exposed
            + (current_stages == self.INFECTED_VAR) * after_infected
        )

        # update curr stage - if exposed at current step t or not
        current_stages = newly_exposed_today * self.EXPOSED_VAR + stage_progression
        return current_stages


def lam(x_i, x_j, edge_attr, t, R, SFSusceptibility, SFInfector, lam_gamma_integrals):
    """
    x_i and x_j are attributes from nodes for all edges in the graph
        note: x_j[:, 2].sum() will be higher than current infections because there are some nodes repeated
    """
    S_A_s = SFSusceptibility[x_i[:, 0].long()]  # age dependant
    A_s_i = SFInfector[x_j[:, 1].long()]  # stage dependant
    B_n = edge_attr[1, :]
    integrals = torch.zeros_like(B_n)
    infected_idx = x_j[:, 2].bool()
    infected_times = t - x_j[infected_idx, 3]

    integrals[infected_idx] = lam_gamma_integrals[
        infected_times.long()
    ]  #:,2 is infected index and :,3 is infected time
    edge_network_numbers = edge_attr[
        0, :
    ]  # to account for the fact that mean interactions start at 4th position of x
    I_bar = torch.gather(x_i[:, 4:27], 1, edge_network_numbers.view(-1, 1).long()).view(-1)
    # to account for the fact that mean interactions start at 4th position of x. how we can explain the above ?
    # let's assume that:
    # x_i =
    #   1   2   3   4
    #   1   2   3   4
    #   1   2   3   4
    # edge_network_numbers = [2, 0, 3], so edge_network_numbers.view(-1, 1).long() =
    #    2
    #    0
    #    3
    # torch.gather(input, dim, index) is a function that gathers elements from input tensor
    # along the specified dim using the indices provided in the index tensor.
    # In our case, x is the input tensor, 1 is the dimension along which we want to gather elements (columns),
    # and edge_network_numbers.view(-1, 1).long() are the indices specifying which elements to gather.
    # In our example, this step gathers the elements from x using the indices [2, 0, 3] along the columns dimension (dim=1):
    # So we have:
    # 3   1   4
    # 1   1   4
    # 3   1   4
    # And then .view(-1) reshapes the tensor into a 1-dimensional tensor, e.g.,
    # 3   1   4   1   1   4   3   1   4

    res = R * S_A_s * A_s_i * B_n * integrals / I_bar  # Edge attribute 1 is B_n

    return res.view(-1, 1)


class InfectionNetwork(MessagePassing):
    """Contact network with graph message passing function for disease spread"""

    def __init__(self, lam, SFSusceptibility, SFInfector, lam_gamma_integrals, device):
        super(InfectionNetwork, self).__init__(aggr="add")
        self.lam = lam
        self.SFSusceptibility = SFSusceptibility
        self.SFInfector = SFInfector
        self.lam_gamma_integrals = lam_gamma_integrals
        self.device = device

    def forward(self, data, r0_value_trainable):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        t = data.t
        return self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            t=t,
            R=r0_value_trainable,
            SFSusceptibility=self.SFSusceptibility,
            SFInfector=self.SFInfector,
            lam_gamma_integrals=self.lam_gamma_integrals,
        )

    def message(
        self, x_i, x_j, edge_attr, t, R, SFSusceptibility, SFInfector, lam_gamma_integrals
    ):
        # x_j has shape [E, in_channels]
        tmp = self.lam(
            x_i, x_j, edge_attr, t, R, SFSusceptibility, SFInfector, lam_gamma_integrals
        )  # tmp has shape [E, 2 * in_channels]
        return tmp


# all_agents, all_interactions, num_steps, num_agents
class GradABM:
    def __init__(self, params, device):
        self.params = params
        self.device = device

        # Getting agents
        agents_df = self.params["all_agents"]
        self.agents_ix = torch.arange(0, len(agents_df)).long().to(self.device)
        self.agents_ages = torch.tensor(agents_df["age_group"].to_numpy()).long().to(self.device)

        self.params["num_agents"] = len(agents_df)
        self.num_agents = self.params["num_agents"]

        # Getting interaction:
        all_interactions = self.params["all_interactions"]
        self.agents_mean_interactions_mu = all_interactions["agents_mean_interactions_mu"]
        self.agents_mean_interactions_mu_split = all_interactions[
            "agents_mean_interactions_mu_split"
        ]
        self.network_type_dict_inv = all_interactions["network_type_dict_inv"]
        self.network_type_dict = all_interactions["network_type_dict"]

        self.DPM = SEIRMProgression(self.params)

        self.SFSusceptibility = (
            torch.tensor(self.params["infection_cfg"]["scaling_factor"]["age_dependant"])
            # torch.tensor([0.1])
            .float().to(self.device)
        )
        # Scale factor for a infector being asym, five stages:
        # - SUSCEPTIBLE_VAR = 0
        # - EXPOSED_VAR = 1  # exposed state
        # - INFECTED_VAR = 2
        # - RECOVERED_VAR = 3
        # - MORTALITY_VAR = 4
        self.SFInfector = (
            torch.tensor(self.params["infection_cfg"]["scaling_factor"]["symptom_dependant"])
            .float()
            .to(self.device)
        )

        self.lam_gamma_integrals = self._get_lam_gamma_integrals(
            self.params["infection_cfg"]["gamma_func"]["a"],
            self.params["infection_cfg"]["gamma_func"]["b"],
            int(self.params["infection_cfg"]["total_infection_days"]),
        )  # add 10 to make sure we cover all
        self.lam_gamma_integrals = self.lam_gamma_integrals.to(self.device)
        self.net = InfectionNetwork(
            lam,
            self.SFSusceptibility,
            self.SFInfector,
            self.lam_gamma_integrals,
            self.device,
        ).to(self.device)

        self.current_time = 0
        self.all_edgelist = all_interactions["all_edgelist"]
        self.all_edgeattr = all_interactions["all_edgeattr"]

    def get_interaction_graph(self, t):
        return self.all_edgelist, self.all_edgeattr

    def init_state_tensors(self, learnable_params):
        # Initalizae current stages
        self.current_stages = self.DPM.init_infected_agents(
            learnable_params["initial_infected_percentage"], self.device
        )
        self.agents_infected_index = (self.current_stages >= self.DPM.INFECTED_VAR).to(self.device)
        self.agents_infected_time = self.DPM.init_infected_time(self.current_stages, self.device)

        self.agents_next_stage_times = self.DPM.init_agents_next_stage_time(
            self.current_stages,
            learnable_params["infected_to_recovered_or_dead_time"],
            self.device,
        )
        self.agents_next_stage_times = self.agents_next_stage_times.float()
        self.agents_infected_time = self.agents_infected_time.float()

    def step(self, t, param_t):
        """Send as input: r0_value [hidden state] -> trainable parameters  and t is the time-step of simulation."""
        # construct dictionary with trainable parameters
        learnable_params = {
            "r0_value": param_t[0],
            "mortality_rate": param_t[1],
            "initial_infected_percentage": param_t[2],
            "exposed_to_infected_time": param_t[3],
            "infected_to_recovered_or_dead_time": param_t[4],
            "infection_gamma_pdf_loc": 0,  # param_t[5],
            "infection_gamma_pdf_a": 1.2,  # param_t[6],
            "infection_gamma_pdf_scale": 0.5,  # param_t[7],
        }

        """ change params that were set in constructor """
        if t == 0:
            self.init_state_tensors(learnable_params)

        # self.agents_next_stage_times = torch.round(self.agents_next_stage_times)
        all_edgelist, all_edgeattr = self.get_interaction_graph(
            t
        )  # the interaction graphs for GNN at time t
        all_nodeattr = torch.stack(
            (
                self.agents_ages,  # 0
                self.current_stages.detach(),  # 1
                self.agents_infected_index.to(self.device),  # 2
                self.agents_infected_time.to(self.device),  # 3
                *self.agents_mean_interactions_mu_split,  # 4 to 26
                torch.arange(self.params["num_agents"]).to(self.device),  # Agent ids (27)
            )
        ).t()
        agents_data = Data(
            all_nodeattr,
            edge_index=all_edgelist,
            edge_attr=all_edgeattr,
            t=t,
            agents_mean_interactions=self.agents_mean_interactions_mu,
        )

        """
        res = [
            (
                gamma.cdf(
                    t_i,
                    a=learnable_params["infection_gamma_pdf_a"].item(),
                    loc=0,
                    scale=learnable_params["infection_gamma_pdf_scale"].item(),
                )
                - gamma.cdf(
                    t_i - 1,
                    a=learnable_params["infection_gamma_pdf_a"].item(),
                    loc=0,
                    scale=learnable_params["infection_gamma_pdf_scale"].item(),
                )
            )
            for t_i in range(20)
        ]
        """
        # self.net.lam_gamma_integrals = torch.tensor(res).float().to(self.device)

        # self.net.lam_gamma_integrals = torch.tensor([0.0]).float().to(self.device)

        lam_t = self.net(agents_data, learnable_params["r0_value"])
        prob_not_infected = torch.exp(-lam_t)
        p = torch.hstack((1 - prob_not_infected, prob_not_infected))
        cat_logits = torch.log(p + 1e-9)
        potentially_exposed_today = F.gumbel_softmax(logits=cat_logits, tau=1, hard=True, dim=1)[
            :, 0
        ]  # first column is prob of infections

        newly_exposed_today = self.DPM.get_newly_exposed(
            self.current_stages, potentially_exposed_today
        )

        if t > 10:
            x = 3

        recovered_dead_now, target2 = self.DPM.get_target_variables(
            learnable_params, self.current_stages, self.agents_next_stage_times, t, self.device
        )
        # print(f"{t}: {self.current_stages.tolist()} ~ {potentially_exposed_today} ~ {target2}")
        recovered_dead_now = torch.round(recovered_dead_now)

        # get next stages without updating yet the current_stages
        next_stages = self.DPM.update_current_stage(
            newly_exposed_today, self.current_stages, self.agents_next_stage_times, t
        )

        # update times with current_stages
        self.agents_next_stage_times = self.DPM.update_next_stage_times(
            learnable_params,
            newly_exposed_today,
            self.current_stages,
            self.agents_next_stage_times,
            t,
        )

        # safely update current_stages
        self.current_stages = next_stages

        # update for newly exposed agents {exposed_today}
        self.agents_infected_index[newly_exposed_today.bool()] = True
        self.agents_infected_time[newly_exposed_today.bool()] = t
        # remove recovered from infected indexes
        self.agents_infected_index[recovered_dead_now.bool()] = False

        # reconcile and return values
        self.current_time += 1

        return None, target2

    def _get_lam_gamma_integrals(self, a, b, t):
        res = [
            (gamma.cdf(t_i, a=a, loc=0, scale=b) - gamma.cdf(t_i - 1, a=a, loc=0, scale=b))
            for t_i in range(t)
        ]
        return torch.tensor(res).float()


def param_model_forward(param_model):
    return param_model.forward()


def forward_simulator(param_values, abm, training_num_steps, devices):
    """assumes abm contains only one simulator for covid (one county), and multiple for flu (multiple counties)"""
    num_counties = 1
    param_values = param_values.squeeze(0)
    predictions = []
    training_num_steps = 15
    for time_step in range(training_num_steps):
        _, pred_t = abm.step(time_step, param_values.to(abm.device))
        pred_t = pred_t.type(torch.float64)
        predictions.append(pred_t.to(devices[0]))
    predictions = torch.stack(predictions, 0).reshape(1, -1)  # num counties, seq len
    predictions = predictions.reshape(num_counties, -1)

    return predictions.unsqueeze(2)


def build_simulator(
    devices, all_agents, all_interactions, infection_cfg: dict, start_num_step: int = 0
):
    """build simulator: ABM or ODE"""
    params = {
        # "num_steps": start_num_step,
        "all_agents": all_agents,
        "all_interactions": all_interactions,
        "infection_cfg": infection_cfg,
    }
    abm = GradABM(params, devices[0])

    return abm
