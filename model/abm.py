from abc import ABC, abstractmethod
from copy import copy as shallow_copy

import torch
import torch.nn.functional as F
from numpy import array
from scipy.stats import gamma as stats_gamma
from torch.distributions import Gamma as torch_gamma
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from model import ALL_PARAMS, MAX_INFECTIOUS_GAMMA_RATE, STAGE_INDEX
from model.utils import round_a_list


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
        # SUSCEPTIBLE (0) => EXPOSED (1) => INFECTED (2) => RECOVERED/MORTALITY (3)
        self.num_agents = params["num_agents"]

    def add_random_infected(
        self,
        random_percentage,
        infected_to_recovered_time,
        agents_stages,
        agents_infected_time,
        agents_next_stage_times,
        t,
        device,
    ):
        # Add random infected:
        random_infected_p = (random_percentage / 100.0) * torch.ones((self.num_agents, 1)).to(
            device
        )
        random_infected_p[:, 0][agents_stages != STAGE_INDEX["susceptible"]] = 0
        p = torch.hstack((random_infected_p, 1 - random_infected_p))
        cat_logits = torch.log(p + 1e-9)
        agents_stages_with_random_infected = F.gumbel_softmax(
            logits=cat_logits, tau=1, dim=1, hard=True
        )[:, 0]
        agents_stages_with_random_infected *= STAGE_INDEX["infected"]
        agents_stages = agents_stages + agents_stages_with_random_infected
        # print(t, agents_stages_with_random_infected.tolist().count(2))
        # Updated infected time:
        agents_infected_time[agents_stages_with_random_infected == STAGE_INDEX["infected"]] = t

        # Updated init_agents_next_stage_time:
        agents_next_stage_times[agents_stages_with_random_infected == STAGE_INDEX["infected"]] = (
            t + infected_to_recovered_time
        )

        return agents_stages, agents_infected_time, agents_next_stage_times

    def init_infected_agents(self, initial_infected_percentage, device):
        prob_infected = (initial_infected_percentage / 100) * torch.ones((self.num_agents, 1)).to(
            device
        )
        p = torch.hstack((prob_infected, 1 - prob_infected))
        cat_logits = torch.log(p + 1e-9)
        agents_stages = F.gumbel_softmax(logits=cat_logits, tau=1, hard=True, dim=1)[:, 0]
        agents_stages *= STAGE_INDEX["infected"]
        agents_stages = agents_stages.to(device)

        return agents_stages

    def init_infected_time(self, agents_stages, device):
        agents_infected_time = -1 * torch.ones_like(agents_stages).to(device)
        agents_infected_time[agents_stages == STAGE_INDEX["infected"]] = 0

        return agents_infected_time

    def init_agents_next_stage_time(
        self, agents_stages, infected_to_recovered_or_dead_time, device
    ):
        agents_next_stage_times = 0.001 * torch.ones_like(agents_stages).long().to(device)
        agents_next_stage_times[agents_stages == STAGE_INDEX["infected"]] = (
            0 + infected_to_recovered_or_dead_time
        )
        return agents_next_stage_times

    def update_initial_times(self, learnable_params, agents_stages, agents_next_stage_times):
        """this is for the abm constructor"""
        infected_to_recovered_time = learnable_params["infected_to_recovered_time"]
        exposed_to_infected_time = learnable_params["exposed_to_infected_time"]
        agents_next_stage_times[agents_stages == STAGE_INDEX["exposed"]] = exposed_to_infected_time
        agents_next_stage_times[
            agents_stages == STAGE_INDEX["infected"]
        ] = infected_to_recovered_time
        return agents_next_stage_times

    def get_newly_exposed(self, current_stages, potentially_exposed_today):
        # we now get the ones that new to exposure
        newly_exposed_today = (
            current_stages == STAGE_INDEX["susceptible"]
        ) * potentially_exposed_today
        return newly_exposed_today

    def update_next_stage_times(
        self,
        exposed_to_infected_time,
        infected_to_recovered_or_dead_time,
        newly_exposed_today,
        current_stages,
        agents_next_stage_times,
        t,
        total_timesteps,
    ):
        new_transition_times = torch.clone(agents_next_stage_times)
        curr_stages = torch.clone(current_stages).long()
        new_transition_times[
            (curr_stages == STAGE_INDEX["infected"]) * (t >= agents_next_stage_times)
        ] = (total_timesteps + 1)
        new_transition_times[
            (curr_stages == STAGE_INDEX["exposed"]) * (t >= agents_next_stage_times)
        ] = (t + infected_to_recovered_or_dead_time)
        return (
            newly_exposed_today * (t + 1 + exposed_to_infected_time)
            + (1 - newly_exposed_today) * new_transition_times
        )

    def get_target_variables(
        self, mortality_rate, current_stages, agents_next_stage_times, t, device
    ):
        recovered_or_dead_today = (
            current_stages
            * (current_stages == STAGE_INDEX["infected"])
            * (t >= agents_next_stage_times)
        ) / STAGE_INDEX[
            "infected"
        ]  # agents when stage changes

        death_total_today = (mortality_rate / 100.0) * torch.sum(recovered_or_dead_today)
        return recovered_or_dead_today, death_total_today

    def update_current_stage(
        self, newly_exposed_today, current_stages, agents_next_stage_times, t
    ):
        """progress disease: move agents to different disease stage"""
        agents_next_stage_times_max = agents_next_stage_times + 1.0

        after_exposed = STAGE_INDEX["exposed"] * (t < agents_next_stage_times) + STAGE_INDEX[
            "infected"
        ] * ((t >= agents_next_stage_times) & (t < agents_next_stage_times_max))

        after_infected = STAGE_INDEX["infected"] * (t < agents_next_stage_times) + STAGE_INDEX[
            "recovered_or_death"
        ] * ((t >= agents_next_stage_times) & (t < agents_next_stage_times_max))

        stage_progression = (
            (current_stages == STAGE_INDEX["susceptible"]) * STAGE_INDEX["susceptible"]
            + (current_stages == STAGE_INDEX["recovered_or_death"])
            * STAGE_INDEX["recovered_or_death"]
            + (current_stages == STAGE_INDEX["exposed"]) * after_exposed
            + (current_stages == STAGE_INDEX["infected"]) * after_infected
        )

        current_stages = newly_exposed_today * STAGE_INDEX["exposed"] + stage_progression
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

    # Value range:
    #  - R: 0.1 - 15.0
    #  - S_A_s: 0.35 - 1.52
    #  - A_s_i: 0.0 - 0.72
    #  - integrals: 1.52
    #  - B_n: 0.1
    #  - I_bar: 2
    res = R * S_A_s * A_s_i * B_n * integrals / I_bar  # Edge attribute 1 is B_n

    return res.view(-1, 1)


class InfectionNetwork(MessagePassing):
    """Contact network with graph message passing function for disease spread"""

    def __init__(self, lam, SFSusceptibility, SFInfector, device):
        super(InfectionNetwork, self).__init__(aggr="add")
        self.lam = lam
        self.SFSusceptibility = SFSusceptibility
        self.SFInfector = SFInfector
        # self.lam_gamma_integrals = lam_gamma_integrals
        self.device = device

    def forward(self, data, r0_value_trainable, lam_gamma_integrals):
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
            lam_gamma_integrals=lam_gamma_integrals,
        )

    def message(
        self,
        x_i,
        x_j,
        edge_attr,
        t,
        R,
        SFSusceptibility,
        SFInfector,
        lam_gamma_integrals,
    ):
        # x_j has shape [E, in_channels]
        tmp = self.lam(
            x_i,
            x_j,
            edge_attr,
            t,
            R,
            SFSusceptibility,
            SFInfector,
            lam_gamma_integrals,
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
        self.SFInfector = (
            torch.tensor(self.params["infection_cfg"]["scaling_factor"]["symptom_dependant"])
            .float()
            .to(self.device)
        )

        # self.lam_gamma_integrals = self._get_lam_gamma_integrals(
        #    self.params["infection_cfg"]["gamma_func"]["shape"],
        #    self.params["infection_cfg"]["gamma_func"]["scale"],
        #    self.params["infection_cfg"]["gamma_func"]["loc"],
        #    int(self.params["infection_cfg"]["gamma_func"]["max_infectiousness_days"]),
        # )  # add 10 to make sure we cover all
        # self.lam_gamma_integrals = self.lam_gamma_integrals.to(self.device)
        self.net = InfectionNetwork(
            lam,
            self.SFSusceptibility,
            self.SFInfector,
            # self.lam_gamma_integrals,
            self.device,
        ).to(self.device)

        self.current_time = 0
        self.all_edgelist = all_interactions["all_edgelist"]
        self.all_edgeattr = all_interactions["all_edgeattr"]

    def get_newly_exposed(self, r0_value, lam_gamma_integrals, t):
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
            edge_index=self.all_edgelist,
            edge_attr=self.all_edgeattr,
            t=t,
            agents_mean_interactions=self.agents_mean_interactions_mu,
        )

        lam_t = self.net(agents_data, r0_value, lam_gamma_integrals)
        prob_not_infected = torch.exp(-lam_t)
        p = torch.hstack((1 - prob_not_infected, prob_not_infected))
        cat_logits = torch.log(p + 1e-9)
        potentially_exposed_today = F.gumbel_softmax(logits=cat_logits, tau=1, hard=True, dim=1)[
            :, 0
        ]  # first column is prob of infections

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
        self, initial_infected_percentage, infected_to_recovered_or_dead_time
    ):
        self.current_stages = self.DPM.init_infected_agents(
            initial_infected_percentage, self.device
        )
        self.agents_infected_index = (self.current_stages >= STAGE_INDEX["infected"]).to(
            self.device
        )
        self.agents_infected_time = self.DPM.init_infected_time(self.current_stages, self.device)

        self.agents_next_stage_times = self.DPM.init_agents_next_stage_time(
            self.current_stages,
            infected_to_recovered_or_dead_time,
            self.device,
        )
        self.agents_next_stage_times = self.agents_next_stage_times.float()
        self.agents_infected_time = self.agents_infected_time.float()

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
        all_params = {}
        for proc_param in ALL_PARAMS:
            if proc_param in param_info["learnable_param_order"]:
                try:
                    all_params[proc_param] = param_t[
                        param_info["learnable_param_order"].index(proc_param)
                    ]
                except IndexError:  # if only one learnable param
                    all_params[proc_param] = param_t
            else:
                all_params[proc_param] = param_info["learnable_param_default"][proc_param]
        return all_params

    def cal_lam_gamma_integrals(
        self, shape, scale, infection_gamma_scaling_factor, max_infectiousness_days=20
    ):
        self.lam_gamma_integrals = self._get_lam_gamma_integrals(
            shape,
            scale,
            infection_gamma_scaling_factor,
            int(max_infectiousness_days),
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
        """Send as input: r0_value [hidden state] -> trainable parameters  and t is the time-step of simulation."""

        proc_params = self.get_params(param_info, param_t)

        if t == 0:
            self.init_infected_tensors(
                proc_params["initial_infected_percentage"],
                proc_params["infected_to_recovered_or_death_time"],
            )
            self.cal_lam_gamma_integrals(
                proc_params["infection_gamma_shape"],
                proc_params["infection_gamma_scale"],
                proc_params["infection_gamma_scaling_factor"],
            )
        else:
            (
                self.current_stages,
                self.agents_infected_time,
                self.agents_next_stage_times,
            ) = self.create_random_infected_tensors(
                proc_params["random_infected_percentgae"],
                proc_params["infected_to_recovered_or_death_time"],
                t,
            )

        newly_exposed_today = self.get_newly_exposed(
            proc_params["r0"], self.lam_gamma_integrals, t
        )

        recovered_dead_now, target2 = self.DPM.get_target_variables(
            proc_params["mortality_rate"],
            self.current_stages,
            self.agents_next_stage_times,
            t,
            self.device,
        )

        stage_records = None
        if save_records:
            stage_records = shallow_copy(self.current_stages.tolist())

        if debug:
            self.print_debug_info(
                self.current_stages, self.agents_next_stage_times, newly_exposed_today, target2, t
            )

        # get next stages without updating yet the current_stages
        next_stages = self.DPM.update_current_stage(
            newly_exposed_today, self.current_stages, self.agents_next_stage_times, t
        )

        # update times with current_stages
        self.agents_next_stage_times = self.DPM.update_next_stage_times(
            proc_params["exposed_to_infected_time"],
            proc_params["infected_to_recovered_or_death_time"],
            newly_exposed_today,
            self.current_stages,
            self.agents_next_stage_times,
            t,
            total_timesteps,
        )

        self.current_stages = next_stages

        # update for newly exposed agents {exposed_today}
        self.agents_infected_index[newly_exposed_today.bool()] = True
        self.agents_infected_time[newly_exposed_today.bool()] = t
        # remove recovered from infected indexes
        self.agents_infected_index[recovered_dead_now.bool()] = False

        # reconcile and return values
        self.current_time += 1

        return stage_records, target2

    def _get_lam_gamma_integrals(self, a, b, infection_gamma_scaling_factor, total_t):
        gamma_dist = torch_gamma(concentration=a, rate=1 / b)

        res = gamma_dist.log_prob(torch.tensor(range(total_t))).exp()

        res_factor = infection_gamma_scaling_factor / max(res.tolist())

        res = res * res_factor

        return res


def param_model_forward(param_model, temporal_ref):
    return param_model.forward()


def forward_simulator(
    param_values_all,
    param_info,
    use_temporal_params,
    abm,
    training_num_steps,
    devices,
    save_records: bool = False,
):
    predictions = []
    all_records = []

    param_values_all = param_values_all.to(abm.device)

    for time_step in range(training_num_steps):
        if use_temporal_params:
            param_values = param_values_all[0, time_step, :].to(abm.device)
        else:
            param_values = param_values_all

        proc_record, pred_t = abm.step(
            time_step,
            param_values,
            param_info,
            training_num_steps,
            debug=False,
            save_records=save_records,
        )

        pred_t = pred_t.type(torch.float64)
        predictions.append(pred_t.to(devices[0]))
        all_records.append(proc_record)

    predictions = torch.stack(predictions, 0).reshape(1, -1)

    if any(item is None for item in all_records):
        all_records = None
    else:
        all_records = array(all_records)

    return {"prediction": predictions, "all_records": all_records}


def build_simulator(
    devices, all_agents, all_interactions, infection_cfg: dict, start_num_step: int = 0
):
    """build simulator: ABM or ODE"""
    params = {
        "all_agents": all_agents,
        "all_interactions": all_interactions,
        "infection_cfg": infection_cfg,
    }
    abm = GradABM(params, devices[0])

    return abm
