from typing import Literal

Country         = Literal['germany','hungary']
AdminLevel      = Literal['nuts1', 'nuts2', 'nuts3']
Disease         = Literal['influenza','chickenpox']
ColumnType      = Literal['context','feature','target','pred','split']
DataSetSplit    = Literal['train','val','test']