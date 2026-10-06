import math

from .hhh4gnnmodel import HHH4GNNModel
from .....dataloading import GraphDataBuilder

from ..modules import HHH4GNNModuleV3


class HHH4GNNModelV3(HHH4GNNModel):
    """
    HHH4-GNN - Model (V3: seasonal rates + GRU rate dynamics)

    Same interface as ``HHH4GNNModel``, plus the GRU settings; see
    ``HHH4GNNModuleV3``.
    """
    _module_class = HHH4GNNModuleV3

    def __init__(self,
                 databuilder: GraphDataBuilder,
                 name:        str = 'hhh4gnnmodel_v3'):
        super().__init__(databuilder=databuilder, name=name)

    def set_model_hparams(self,
                          node_penalty:       float            = 0.01,
                          case_feature:       str | None       = None,
                          endemic_features:   list[str] | None = None,
                          dynamics_hidden:    int              = 16,
                          max_log_rate_shift: float            = math.log(20.0),
                          dynamics_penalty:   float            = 1e-3):
        """
        Parameters
        ----------
        node_penalty, case_feature, endemic_features
            As in ``HHH4GNNModel.set_model_hparams``.
        dynamics_hidden : int
            Hidden size of the GRU.
        max_log_rate_shift : float
            Bound on the GRU's shift of the log-rates (log 20: at most 20x either way).
        dynamics_penalty : float
            Ridge penalty on the shifts.
        """
        super().set_model_hparams(node_penalty       = node_penalty,
                                  case_feature       = case_feature,
                                  endemic_features   = endemic_features,
                                  dynamics_hidden    = dynamics_hidden,
                                  max_log_rate_shift = max_log_rate_shift,
                                  dynamics_penalty   = dynamics_penalty)
