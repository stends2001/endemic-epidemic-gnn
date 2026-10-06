import math

import torch
import torch.nn as nn

from .hhh4gnnmodulev2 import HHH4GNNModuleV2


class HHH4GNNModuleV3(HHH4GNNModuleV2):
    """
    HHH4-GNN - Module (V3: seasonal rates + GRU rate dynamics)

    As V2, plus a shift of both log-rates that follows the recent course of the
    epidemic in each region:

        log lambda_it = beta  + b_i + g_lambda' z_t + r_lambda_it
        log phi_it    = gamma + d_i + g_phi'    z_t + r_phi_it

        (r_lambda_it, r_phi_it) = M * tanh(W h_it + c)

    ``h_it`` is the last hidden state of a GRU that reads, for every week of the
    input window, log(1 + own counts), log(1 + neighbour mean) and their change
    from the week before. The GRU runs over the window only (fresh hidden state
    every forward call), so snapshots stay independent. ``M`` bounds the shift
    (default log 20: rates scaled by at most 20x either way). ``W`` and ``c``
    start at zero, so training starts from V2.

    Extra parameters
    ----------------
    dynamics_hidden: int
        hidden size of the GRU
    max_log_rate_shift: float
        bound M on the shift
    dynamics_penalty: float
        ridge penalty on the shifts r (keeps them small unless the data need them)
    """
    def __init__(self, *args,
                 dynamics_hidden:    int   = 16,
                 max_log_rate_shift: float = math.log(20.0),
                 dynamics_penalty:   float = 1e-3,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.max_log_rate_shift = max_log_rate_shift
        self.dynamics_penalty   = dynamics_penalty

        self.gru  = nn.GRU(input_size=4, hidden_size=dynamics_hidden, batch_first=True)
        self.head = nn.Linear(dynamics_hidden, 2 * self.horizon_size)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
        self._last_shift = None

    def _rate_shifts(self, z: torch.Tensor, y: torch.Tensor, ybar: torch.Tensor):
        epi_seasonal, ne_seasonal = super()._rate_shifts(z, y, ybar)

        lo, ln = torch.log1p(y), torch.log1p(ybar)                                      # [N, S]
        d_lo = torch.diff(lo, dim=-1, prepend=lo[:, :1])                                 # weekly change
        d_ln = torch.diff(ln, dim=-1, prepend=ln[:, :1])
        seq  = torch.stack([lo, ln, d_lo, d_ln], dim=-1)                                 # [N, S, 4]

        out, _ = self.gru(seq)                                                           # fresh state per call
        shift  = self.max_log_rate_shift * torch.tanh(self.head(out[:, -1]))            # [N, 2H]
        self._last_shift = shift

        H = self.horizon_size
        return epi_seasonal + shift[:, :H], ne_seasonal + shift[:, H:]

    def regularization(self) -> torch.Tensor:
        total = super().regularization()
        if self._last_shift is not None and self.dynamics_penalty > 0:
            total = total + self.dynamics_penalty * self._last_shift.pow(2).mean()
        return total
