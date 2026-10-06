import torch
import torch.nn as nn

from .hhh4gnnmodule import HHH4GNNModule


class HHH4GNNModuleV2(HHH4GNNModule):
    """
    HHH4-GNN - Module (V2: seasonal rates)

    As V1, but the epidemic and neighbourhood rates vary with the season, as in
    hhh4:

        log lambda_it = beta  + b_i + g_lambda' z_t
        log phi_it    = gamma + d_i + g_phi'    z_t

    ``z_t`` are the seasonal features at the last week of the window (the same
    features as in the endemic branch, e.g. sin / cos of the week). The
    coefficients start at zero, so training starts from V1.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_season = len(self.season_idx)
        self.epi_season = nn.Parameter(torch.zeros(n_season, self.horizon_size))
        self.ne_season  = nn.Parameter(torch.zeros(n_season, self.horizon_size))

    def _rate_shifts(self, z: torch.Tensor, y: torch.Tensor, ybar: torch.Tensor):
        return z @ self.epi_season, z @ self.ne_season
