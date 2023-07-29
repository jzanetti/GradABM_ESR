import torch.nn.functional as F
from numpy import array, isnan, where
from numpy.random import choice
from torch import clamp as torch_clamp
from torch import clone as torch_clone
from torch import eq as torch_eq
from torch import hstack as torch_hstack
from torch import log as torch_log
from torch import manual_seed as torch_seed
from torch import masked_select as torch_masked_select
from torch import ones as torch_ones
from torch import ones_like as torch_ones_like
from torch import sum as torch_sum
from torch import zeros_like as torch_zeros_like

from model import SMALL_VALUE, STAGE_INDEX, TORCH_SEED_NUM
from model.disease_progression import DiseaseProgression


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
        random_infected_p = (random_percentage / 100.0) * torch_ones((self.num_agents, 1)).to(
            device
        )
        random_infected_p[:, 0][agents_stages != STAGE_INDEX["susceptible"]] = 0
        p = torch_hstack((random_infected_p, 1 - random_infected_p))
        cat_logits = torch_log(p + 1e-9)

        while True:
            if TORCH_SEED_NUM is not None:
                torch_seed(TORCH_SEED_NUM["random_infected"])
            agents_stages_with_random_infected = F.gumbel_softmax(
                logits=cat_logits, tau=1, dim=1, hard=True
            )[:, 0]

            if not isnan(agents_stages_with_random_infected.cpu().clone().detach().numpy()).any():
                break

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
        prob_infected = (initial_infected_percentage / 100) * torch_ones((self.num_agents, 1)).to(
            device
        )
        p = torch_hstack((prob_infected, 1 - prob_infected))
        cat_logits = torch_log(p + 1e-9)
        if TORCH_SEED_NUM is not None:
            torch_seed(TORCH_SEED_NUM["initial_infected"])
        agents_stages = F.gumbel_softmax(logits=cat_logits, tau=1, hard=True, dim=1)[:, 0]
        agents_stages *= STAGE_INDEX["infected"]
        agents_stages = agents_stages.to(device)

        return agents_stages

    def init_infected_time(self, agents_stages, device):
        agents_infected_time = -1 * torch_ones_like(agents_stages).to(device)
        agents_infected_time[agents_stages == STAGE_INDEX["infected"]] = 0

        return agents_infected_time

    def init_agents_next_stage_time(
        self, agents_stages, infected_to_recovered_or_dead_time, device
    ):
        agents_next_stage_times = 0.001 * torch_ones_like(agents_stages).long().to(device)
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
        # susceptible_mask = self.create_stage_mask(
        #    current_stages, STAGE_INDEX["susceptible"], bias=1.0
        # )

        # newly_exposed_today = susceptible_mask * potentially_exposed_today
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
        new_transition_times = torch_clone(agents_next_stage_times)
        curr_stages = torch_clone(current_stages).long()
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
        self,
        mortality_rate,
        current_stages,
        agents_next_stage_times,
        infected_to_recovered_or_death_time,
        t,
        apply_sigmoid=False,
    ):
        def _randomly_assign_death_people(recovered_or_dead_today, death_total_today):
            recovered_or_dead_today_array = array(recovered_or_dead_today.tolist())
            indices_ones = where(recovered_or_dead_today_array == 1)[0]
            n = min(int(round(death_total_today.item(), 0)), len(indices_ones))
            death_indices = choice(indices_ones, n, replace=False)

            return death_indices

        def _apply_sigmoid_to_mask_t(t, agents_next_stage_times_min, agents_next_stage_times_max):
            import torch

            return torch.sigmoid(10 * (torch.tensor(float(t)) - agents_next_stage_times_min)) * (
                1 - torch.sigmoid(10 * (torch.tensor(float(t)) - agents_next_stage_times_max))
            )

        agents_next_stage_times_max = agents_next_stage_times + 1.0

        if (
            t < infected_to_recovered_or_death_time
        ):  # make sure we have recovered_or_dead_today from t0
            agents_next_stage_times_min = torch_clamp(
                agents_next_stage_times - infected_to_recovered_or_death_time, min=0.0
            )
        else:
            agents_next_stage_times_min = torch_clamp(agents_next_stage_times - 0.5, min=0.0)

        recovered_or_dead_today = (
            current_stages
            * (current_stages == STAGE_INDEX["infected"])
            * ((t >= agents_next_stage_times_min) & (t < agents_next_stage_times_max))
        ) / STAGE_INDEX["infected"]

        # print(t, recovered_or_dead_today.sum())

        if apply_sigmoid:
            recovered_or_dead_today = torch_clamp(
                recovered_or_dead_today, min=SMALL_VALUE
            ) * _apply_sigmoid_to_mask_t(
                t, agents_next_stage_times_min, agents_next_stage_times_max
            )

        death_total_today = (mortality_rate / 100.0) * torch_sum(
            recovered_or_dead_today
        ) + SMALL_VALUE

        death_indices = _randomly_assign_death_people(recovered_or_dead_today, death_total_today)

        return recovered_or_dead_today, death_indices, death_total_today

    def create_stage_mask(self, cur_stage, cur_mask, bias: float = 0.0):
        cur_mask = torch_eq(cur_stage, cur_mask)
        selected_elements = torch_masked_select(cur_stage, cur_mask)
        selected_elements += bias

        output = torch_zeros_like(cur_stage)
        output[cur_mask] = selected_elements
        return output

    def update_current_stage(
        self, newly_exposed_today, current_stages, agents_next_stage_times, death_indices, t
    ):
        """progress disease: move agents to different disease stage"""

        # mask = torch.eq(self.current_stages, STAGE_INDEX["susceptible"])
        # selected_elements = torch.masked_select(self.current_stages, mask)

        # x = torch.zeros_like(self.current_stages)
        # x[mask] = selected_elements

        agents_next_stage_times_max = agents_next_stage_times + 0.5

        after_exposed = STAGE_INDEX["exposed"] * (t < agents_next_stage_times) + STAGE_INDEX[
            "infected"
        ] * ((t >= agents_next_stage_times) & (t < agents_next_stage_times_max))

        after_infected = STAGE_INDEX["infected"] * (t < agents_next_stage_times) + STAGE_INDEX[
            "recovered_or_death"
        ] * ((t >= agents_next_stage_times) & (t < agents_next_stage_times_max))

        after_infected[death_indices] = STAGE_INDEX["death"]

        stage_progression = (
            (current_stages == STAGE_INDEX["susceptible"]) * STAGE_INDEX["susceptible"]
            + (current_stages == STAGE_INDEX["recovered_or_death"])
            * STAGE_INDEX["recovered_or_death"]
            + (current_stages == STAGE_INDEX["exposed"]) * after_exposed
            + (current_stages == STAGE_INDEX["infected"]) * after_infected
        )

        current_stages = newly_exposed_today * STAGE_INDEX["exposed"] + stage_progression

        return current_stages
