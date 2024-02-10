from logging import getLogger
from random import uniform as random_uniform

from numpy import intersect1d as numpy_intersect1d
from numpy import isin as numpy_isin
from numpy import ones as numpy_ones
from numpy import where as numpy_where
from torch import tensor as torch_tensor

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
