from numpy import intersect1d as numpy_intersect1d
from numpy import isin as numpy_isin
from numpy import ones as numpy_ones
from numpy import where as numpy_where
from torch import nonzero as torch_nonzero
from torch import ones_like as torch_ones_like
from torch import randperm as torch_randperm
from torch import tensor as torch_tensor
from torch import zeros_like as torch_zeros_like

from process import DEVICE, LOC_INDEX


def school_closure(infected_idx, edge_attr, school_closure_cfg):
    school_closure_sf = numpy_ones(len(infected_idx))

    if not school_closure_cfg["enable"]:
        return torch_tensor(school_closure_sf).to(DEVICE)

    infected_idx_index = numpy_where(infected_idx.cpu().detach().numpy() == True)[0]

    if len(infected_idx_index) > 0:
        school_index = numpy_where(
            edge_attr[0, :].cpu().detach().numpy() == LOC_INDEX["school"]
        )[0]

        overlap_index = numpy_intersect1d(infected_idx_index, school_index)
        schools_to_shutdown = edge_attr[2, :][overlap_index].unique().cpu().detach()

        potential_indices = numpy_where(
            numpy_isin(edge_attr[2].cpu().detach().numpy(), schools_to_shutdown)
        )[0]
        schools_to_shutdown_indices = numpy_intersect1d(school_index, potential_indices)
        school_closure_sf[schools_to_shutdown_indices] *= school_closure_cfg[
            "scaling_factor"
        ]

    return torch_tensor(school_closure_sf).to(DEVICE)


def infected_case_isolation(
    infected_idx,
    contact_tracing_coverage,
    isolation_compliance_rate,
    isolation_intensity,
    t,
    min_cases: int = 10,
):
    """Create infected case isolation scaling factor

    Args:
        infected_idx: Infected case index

    Returns:
        _type_: Scaling factor for infected case
    """
    if (
        (isolation_compliance_rate is None)
        or (contact_tracing_coverage is None)
        or (t == 0)
    ):
        return torch_ones_like(infected_idx)

    # infected agents (0.0: uninfected; 1.0: infected)
    infected_agents = infected_idx.float()
    infected_agents_index = torch_nonzero(infected_agents == 1.0).squeeze()
    try:
        infected_agents_length = len(infected_agents_index)
    except TypeError:  # len() of a 0-d tensor
        infected_agents_length = -999.0

    if infected_agents_length < min_cases:
        return torch_ones_like(infected_agents)

    identified_infected_agents_length = (
        infected_agents_length * contact_tracing_coverage
    )
    isolated_agents_length = int(
        isolation_compliance_rate * identified_infected_agents_length
    )

    isolated_agents_index = torch_randperm(infected_agents_length)[
        :isolated_agents_length
    ]
    isolated_mask = torch_zeros_like(infected_agents)
    isolated_mask[infected_agents_index[isolated_agents_index]] = (
        1.0 - isolation_intensity
    )

    isolated_sf = 1.0 - isolated_mask

    return isolated_sf
