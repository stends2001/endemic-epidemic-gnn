from .hhh4gnnmodel import HHH4GNNModel
from .....dataloading import GraphDataBuilder

from ..modules import HHH4GNNModuleV2


class HHH4GNNModelV2(HHH4GNNModel):
    """
    HHH4-GNN - Model (V2: seasonal epidemic and neighbourhood rates)

    Same interface as ``HHH4GNNModel``; see ``HHH4GNNModuleV2``.
    """
    _module_class = HHH4GNNModuleV2

    def __init__(self,
                 databuilder: GraphDataBuilder,
                 name:        str = 'hhh4gnnmodel_v2'):
        super().__init__(databuilder=databuilder, name=name)
