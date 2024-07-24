from abc import ABC, abstractmethod

import torch.nn.functional as F
from numpy import arange as numpy_arange
from numpy import array
from numpy import isin as numpy_isin
from numpy import isnan as numpy_isnan
from numpy import sum as numpy_sum
from numpy import where
from numpy import where as numpy_where
from numpy.random import choice as numpy_choice
from numpy.random import random as numpy_random
from numpy.random import seed as numpy_seed
from torch import clone as torch_clone
from torch import eq as torch_eq
from torch import hstack as torch_hstack
from torch import log as torch_log
from torch import long as torch_long
from torch import manual_seed
from torch import masked_select as torch_masked_select
from torch import ones as torch_ones
from torch import ones_like as torch_ones_like
from torch import sum as torch_sum
from torch import tensor as torch_tensor
from torch import where as torch_where
from torch import zeros as torch_zeros
from torch import zeros_like as torch_zeros_like

from process import DEVICE
from process.model import MISSING_DATA, STAGE_INDEX
from process.model.utils import apply_gumbel_softmax


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


class Progression_model(DiseaseProgression):
    """SEIRM for COVID-19"""

    def __init__(self, num_agents: int):
        super(DiseaseProgression, self).__init__()
        # encoding of stages
        # Stage progress:
        # SUSCEPTIBLE (0) => EXPOSED (1) => INFECTED (2) => RECOVERED/MORTALITY (3)
        self.num_agents = num_agents

    def add_random_infected(
        self,
        random_infected_ids,
        random_percentage,
        infected_to_recovered_time,
        agents_stages,
        agents_infected_time,
        agents_next_stage_times,
        t,
        # device,
    ):
        # Add random infected:
        if random_infected_ids is None:
            random_infected_p = (random_percentage / 100.0) * torch_ones(
                (self.num_agents, 1)
            ).to(DEVICE)
            random_infected_p[:, 0][agents_stages != STAGE_INDEX["susceptible"]] = 0
        else:
            random_infected_p = torch_zeros((self.num_agents, 1)).to(DEVICE)
            random_infected_ids_filter = torch_tensor(
                [i in random_infected_ids[t] for i in range(len(agents_stages))]
            ).to(DEVICE)
            random_infected_p[:, 0][
                (agents_stages == STAGE_INDEX["susceptible"])
                & (random_infected_ids_filter)
            ] = 1.0

        p = torch_hstack((random_infected_p, 1 - random_infected_p))
        cat_logits = torch_log(p + 1e-9)

        while True:

            agents_stages_with_random_infected = F.gumbel_softmax(
                logits=cat_logits, tau=1, dim=1, hard=True
            )[:, 0]

            if not numpy_isnan(
                agents_stages_with_random_infected.cpu().clone().detach().numpy()
            ).any():
                break

        agents_stages_with_random_infected *= STAGE_INDEX["infected"]
        agents_stages = agents_stages + agents_stages_with_random_infected

        # print(t, agents_stages_with_random_infected.tolist().count(2))
        # Updated infected time:
        agents_infected_time[
            agents_stages_with_random_infected == STAGE_INDEX["infected"]
        ] = t

        # Updated init_agents_next_stage_time:
        agents_next_stage_times[
            agents_stages_with_random_infected == STAGE_INDEX["infected"]
        ] = (t + infected_to_recovered_time)

        return agents_stages, agents_infected_time, agents_next_stage_times

    def init_infected_agents(
        self,
        initial_infected_ids,
        initial_infected_percentage,
        agents_id,
    ):
        if initial_infected_ids is not None:
            agents_stages = torch_zeros((self.num_agents))
            all_agents_ids = array(agents_id.tolist())
            indices = numpy_where(numpy_isin(all_agents_ids, initial_infected_ids))[0]
            agents_stages[indices] = STAGE_INDEX["infected"]
        else:
            initial_infected_percentage = (
                initial_infected_percentage / 100
            ) * torch_ones((self.num_agents, 1)).to(DEVICE)

            agents_stages = apply_gumbel_softmax(initial_infected_percentage)
            agents_stages *= STAGE_INDEX["infected"]

            agents_stages = torch_where(
                agents_stages == 0, STAGE_INDEX["susceptible"], agents_stages
            )

        return agents_stages.to(DEVICE)

    def init_infected_time(self, agents_stages):
        agents_infected_time = MISSING_DATA * torch_ones_like(agents_stages).to(DEVICE)
        agents_infected_time[agents_stages == STAGE_INDEX["infected"]] = 0
        return agents_infected_time

    def init_agents_next_stage_time(
        self, agents_stages, infected_to_recovered_or_dead_time
    ):
        agents_next_stage_times = torch_zeros_like(agents_stages).long().to(DEVICE)
        agents_next_stage_times[agents_stages == STAGE_INDEX["infected"]] = (
            infected_to_recovered_or_dead_time
        )
        return agents_next_stage_times

    def get_newly_exposed(self, current_stages, potentially_exposed_today):
        # we now get the ones that new to exposure
        newly_exposed_mask = (
            current_stages == STAGE_INDEX["susceptible"]
        ) * potentially_exposed_today

        return newly_exposed_mask

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
            (curr_stages == STAGE_INDEX["infected"]) * (t == agents_next_stage_times)
        ] = (total_timesteps + 1)

        new_transition_times[
            (curr_stages == STAGE_INDEX["exposed"]) * (t == agents_next_stage_times)
        ] = (t + infected_to_recovered_or_dead_time)
        return (
            newly_exposed_today * (t + exposed_to_infected_time)
            + (1 - newly_exposed_today) * new_transition_times
        )

    def _generate_one_hot_tensor(self, timestep, num_timesteps):
        timestep_tensor = torch_tensor([timestep])
        one_hot_tensor = F.one_hot(timestep_tensor, num_classes=num_timesteps)

        return one_hot_tensor.to(DEVICE)

    def get_target_variables(
        self, vaccine_efficiency_symptom, current_stages, cur_target, t, total_timesteps
    ):

        recovered_and_dead_mask = current_stages == STAGE_INDEX["infected"]

        infected_today = (
            vaccine_efficiency_symptom
            * current_stages
            * recovered_and_dead_mask
            / STAGE_INDEX["infected"]
        )
        infected_today = torch_sum(infected_today)

        cur_target = (
            cur_target
            + self._generate_one_hot_tensor(t, total_timesteps) * infected_today
        )

        return cur_target

    def create_stage_mask(self, cur_stage, cur_mask, bias: float = 0.0):
        cur_mask = torch_eq(cur_stage, cur_mask)
        selected_elements = torch_masked_select(cur_stage, cur_mask)
        selected_elements += bias

        output = torch_zeros_like(cur_stage)
        output[cur_mask] = selected_elements
        return output

    def update_current_stage(
        self,
        newly_exposed_mask,
        current_stages,
        agents_next_stage_times,
        t,
        new_method: bool = True,
    ):
        """progress disease: move agents to different disease stage"""

        if new_method:

            # ----------------------
            # Adding susceptible
            # ----------------------
            existing_susceptible_mask = (
                current_stages == STAGE_INDEX["susceptible"]
            ) * (newly_exposed_mask == 0)
            existing_susceptible = current_stages * existing_susceptible_mask

            # ----------------------
            # Adding exposed
            # ----------------------
            newly_exposed = (
                current_stages
                * newly_exposed_mask
                * (STAGE_INDEX["exposed"] / STAGE_INDEX["susceptible"])
            )

            existing_exposed_mask = (current_stages == STAGE_INDEX["exposed"]) * (
                agents_next_stage_times > t
            )
            existing_exposed = current_stages * existing_exposed_mask

            # ----------------------
            # Adding infected
            # ----------------------
            newly_infected_mask = (current_stages == STAGE_INDEX["exposed"]) * (
                agents_next_stage_times == t
            )
            newly_infected = (
                current_stages
                * newly_infected_mask
                * (STAGE_INDEX["infected"] / STAGE_INDEX["exposed"])
            )

            existing_infected_mask = (current_stages == STAGE_INDEX["infected"]) * (
                agents_next_stage_times > t
            )
            existing_infected = current_stages * existing_infected_mask

            # ----------------------
            # Adding removed
            # ----------------------
            newly_removed_mask = (current_stages == STAGE_INDEX["infected"]) * (
                agents_next_stage_times == t
            )
            newly_removed = (
                current_stages
                * newly_removed_mask
                * (STAGE_INDEX["recovered_or_death"] / STAGE_INDEX["infected"])
            )

            existing_removed_mask = (
                current_stages == STAGE_INDEX["recovered_or_death"]
            ) * (agents_next_stage_times > t)
            existing_removed = current_stages * existing_removed_mask

            # ----------------------
            # Adding all of them together
            # ----------------------
            current_stages = (
                existing_susceptible
                + newly_exposed
                + existing_exposed
                + newly_infected
                + existing_infected
                + newly_removed
                + existing_removed
            )
        else:

            after_exposed = STAGE_INDEX["exposed"] * (
                t < agents_next_stage_times
            ) + STAGE_INDEX["infected"] * (t == agents_next_stage_times)

            after_infected = STAGE_INDEX["infected"] * (
                t < agents_next_stage_times
            ) + STAGE_INDEX["recovered_or_death"] * (t == agents_next_stage_times)

            stage_progression = (
                (current_stages == STAGE_INDEX["susceptible"])
                * STAGE_INDEX["susceptible"]
                + (current_stages == STAGE_INDEX["recovered_or_death"])
                * STAGE_INDEX["recovered_or_death"]
                + (current_stages == STAGE_INDEX["exposed"]) * after_exposed
                + (current_stages == STAGE_INDEX["infected"]) * after_infected
            )
            """
            from process.model.utils import soft_equals, soft_less_than

            after_exposed = STAGE_INDEX["exposed"] * soft_less_than(
                t, agents_next_stage_times
            ) + STAGE_INDEX["infected"] * soft_equals(t, agents_next_stage_times)

            after_infected = STAGE_INDEX["infected"] * soft_less_than(
                t, agents_next_stage_times
            ) + STAGE_INDEX["recovered_or_death"] * soft_equals(
                t, agents_next_stage_times
            )

            stage_progression = (
                soft_equals(current_stages, STAGE_INDEX["susceptible"])
                * STAGE_INDEX["susceptible"]
                + soft_equals(current_stages, STAGE_INDEX["recovered_or_death"])
                * STAGE_INDEX["recovered_or_death"]
                + soft_equals(current_stages, STAGE_INDEX["exposed"]) * after_exposed
                + soft_equals(current_stages, STAGE_INDEX["infected"]) * after_infected
            )
            """
            current_stages = (
                newly_exposed_today * STAGE_INDEX["exposed"] + stage_progression
            )

        return current_stages
