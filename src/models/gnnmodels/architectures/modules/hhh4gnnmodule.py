import math

import torch
import torch.nn as nn
from torch_geometric.utils import remove_self_loops, scatter


class HHH4GNNModule(nn.Module):
    """
    HHH4-GNN - Module (V1: constant rates)

    The endemic-epidemic model (hhh4) as a torch module. For every region i it
    predicts the NB mean of the count ``horizon_size`` steps after the input
    window as the sum of three branches:

        mu_i = nu_i                                   (endemic)
             + lambda_i * sum_k w_k   y_i,t-k         (epidemic: own region)
             + phi_i    * sum_k w'_k  ybar_i,t-k      (neighbourhood)

        log nu_i     = a_i + c' z_t        (region intercept + seasonal features)
        log lambda_i = beta  + b_i         (b_i: centred region effect)
        log phi_i    = gamma + d_i         (d_i: centred region effect)

    ``ybar`` is the mean count of the neighbours (edges j -> i), ``w`` / ``w'``
    are learned lag weights over the input window (softmax), and the
    dispersion ``alpha_i`` is one parameter per region (Var = mu + alpha mu^2).

    Parameters
    ----------
    num_nodes: int
        number of regions
    seq_length: int
        number of weeks in the input window
    horizon_size: int
        number of steps to predict
    case_idx: int
        position of the raw case-count feature on the feature axis of x
    season_idx: list[int]
        positions of the seasonal features (e.g. sin / cos of the week)
    mean_count: float
        mean training count; used only to start each branch at a third of it
    node_penalty: float
        ridge penalty on the centred region effects b_i, d_i

    Forward
    -------
    x: [num_nodes, num_features, seq_length]; returns (mu, alpha), each
    [num_nodes, horizon_size]. With ``return_components=True`` also a dict
    with the three branches.
    """
    component_names = ('endemic', 'epidemic', 'neighbourhood')

    def __init__(self,
                 num_nodes:     int,
                 seq_length:    int,
                 horizon_size:  int,
                 case_idx:      int,
                 season_idx:    list[int],
                 mean_count:    float = 1.0,
                 node_penalty:  float = 0.01):

        super().__init__()

        ### set data - params ###
        self.num_nodes      = num_nodes
        self.seq_length     = seq_length
        self.horizon_size   = horizon_size
        self.case_idx       = case_idx
        self.node_penalty   = node_penalty
        self.register_buffer('season_idx', torch.tensor(season_idx, dtype=torch.long))

        third = math.log(max(mean_count, 1e-3) / 3.0)   # each branch starts at ~1/3 of the mean

        # endemic: region intercepts and shared seasonal coefficients, per horizon
        self.end_intercept  = nn.Parameter(torch.full((num_nodes, horizon_size), third))
        self.end_season     = nn.Parameter(torch.zeros(len(season_idx), horizon_size))

        # epidemic: global log-rate, centred region effects, lag weights
        self.epi_log_rate   = nn.Parameter(torch.full((horizon_size,), math.log(1 / 3)))
        self.epi_node       = nn.Parameter(torch.zeros(num_nodes))
        self.epi_lag_logits = nn.Parameter(torch.zeros(horizon_size, seq_length))

        # neighbourhood: same structure, applied to the neighbour mean
        self.ne_log_rate    = nn.Parameter(torch.full((horizon_size,), math.log(1 / 3)))
        self.ne_node        = nn.Parameter(torch.zeros(num_nodes))
        self.ne_lag_logits  = nn.Parameter(torch.zeros(horizon_size, seq_length))

        # dispersion per region
        self.log_alpha      = nn.Parameter(torch.full((num_nodes,), math.log(0.1)))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _centred(p: torch.Tensor) -> torch.Tensor:
        """region effects with mean zero: exp(effect) is relative to the typical region"""
        return p - p.mean()

    @staticmethod
    def _neighbour_mean(y: torch.Tensor, edge_index: torch.Tensor,
                        edge_weight: torch.Tensor | None) -> torch.Tensor:
        """edge-weighted mean over the neighbours j of each region i (edges j -> i): [N, S]"""
        ei, ew = remove_self_loops(edge_index, edge_weight)
        src, dst = ei[0], ei[1]
        w = ew if ew is not None else torch.ones(src.shape[0], device=y.device)
        num = scatter(y[src] * w.unsqueeze(-1), dst, dim=0, dim_size=y.shape[0], reduce='sum')
        den = scatter(w, dst, dim=0, dim_size=y.shape[0], reduce='sum').clamp(min=1e-12)
        return num / den.unsqueeze(-1)

    # ------------------------------------------------------------------ #
    def forward(self,
                x:                  torch.Tensor,
                edge_index:         torch.Tensor,
                edge_weight:        torch.Tensor | None = None,
                return_components:  bool = False):

        y    = x[:, self.case_idx, :].clamp(min=0)               # own counts       [N, S]
        ybar = self._neighbour_mean(y, edge_index, edge_weight)    # neighbour counts [N, S]
        z    = x[:, self.season_idx, -1]                           # season at t      [N, F_s]

        # endemic
        endemic = torch.exp(self.end_intercept + z @ self.end_season)                    # [N, H]

        # time-varying shifts of the log-rates (zero in V1; V2 / V3 override)
        epi_shift, ne_shift = self._rate_shifts(z, y, ybar)                             # [N, H] each

        # epidemic: rate x lag-weighted own counts
        own      = y @ torch.softmax(self.epi_lag_logits, dim=-1).t()                    # [N, H]
        lam      = torch.exp(self.epi_log_rate.view(1, -1) + self._centred(self.epi_node).view(-1, 1) + epi_shift)
        epidemic = lam * own

        # neighbourhood: rate x lag-weighted neighbour counts
        nb            = ybar @ torch.softmax(self.ne_lag_logits, dim=-1).t()             # [N, H]
        phi           = torch.exp(self.ne_log_rate.view(1, -1) + self._centred(self.ne_node).view(-1, 1) + ne_shift)
        neighbourhood = phi * nb

        # kept for inspection: rate multipliers exp(shift) of the latest forward pass
        self.last_rate_multipliers = (torch.exp(epi_shift).detach(), torch.exp(ne_shift).detach())

        mu    = (endemic + epidemic + neighbourhood).clamp(min=1e-6)
        alpha = torch.exp(self.log_alpha).view(-1, 1).expand_as(mu)

        if return_components:
            return (mu, alpha), {'endemic': endemic, 'epidemic': epidemic, 'neighbourhood': neighbourhood}
        return mu, alpha

    def _rate_shifts(self, z: torch.Tensor, y: torch.Tensor, ybar: torch.Tensor):
        """
        Shifts of the epidemic and neighbourhood log-rates, [N, H] each.
        V1: none, the rates are constant over time.
        """
        zero = y.new_zeros(y.shape[0], self.horizon_size)
        return zero, zero

    def regularization(self) -> torch.Tensor:
        """ridge on the centred region effects; added to the loss by Strategy.training_step"""
        return self.node_penalty * (self._centred(self.epi_node).pow(2).mean()
                                    + self._centred(self.ne_node).pow(2).mean())
