import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...utils import Strategy
from ...gnnmodel import GNNModel
from .....dataloading import GraphDataBuilder
from .....utils.types import DataSetSplit

from ..modules import HHH4GNNModule

# one colour per branch, used in every decomposition figure
COMPONENT_COLORS = {'endemic': '#2a78d6', 'epidemic': '#eb6834', 'neighbourhood': '#1baf7a'}


class HHH4GNNModel(GNNModel):
    """
    HHH4-GNN - Model

    Endemic-epidemic (hhh4) model on the graph: the NB mean is the sum of an
    endemic, an epidemic (own region) and a neighbourhood branch, see
    ``HHH4GNNModule``. Output head is always ``'nb'``: forecasts are exact NB
    quantiles (interval mode) or the NB mean (point mode); train with ``loss='nb'``.

    Needs the target and the lag features as raw case counts:
    ``EpiConfig(target_column='cases', lag_column='cases')``.
    """
    _expected_databuilder = 'GraphDataBuilder'

    def __init__(self,
                 databuilder: GraphDataBuilder,
                 name:        str = 'hhh4gnnmodel'):

        super().__init__(
            databuilder = databuilder,
            name        = name,
            strategy    = Strategy()
        )

    def set_model_hparams(self,
                          node_penalty:     float           = 0.01,
                          case_feature:     str | None      = None,
                          endemic_features: list[str] | None = None):
        """
        Parameters
        ----------
        node_penalty : float
            Ridge penalty on the centred region effects of the epidemic and
            neighbourhood rates (added to the training loss).
        case_feature : str | None
            Feature holding the raw case counts of the input window. Default:
            ``'<lag_column>_lag0'``, e.g. ``'cases_lag0'``.
        endemic_features : list[str] | None
            Features entering the endemic branch (log-linear). Default: every
            other feature, e.g. ``['tt_sin_w', 'tt_cos_w']``.
        """
        self._check_counts()
        self._set_output_head('nb')

        feature_names = self.column_registration.get_entries_names_by_type('feature')
        case_feature  = case_feature or f'{self.epiconfig.lag_column}_lag0'
        if case_feature not in feature_names:
            raise ValueError(f'case feature {case_feature!r} not found. Features (in tensor order): {feature_names}')
        if endemic_features is None:
            endemic_features = [f for f in feature_names if f != case_feature]
        missing = [f for f in endemic_features if f not in feature_names]
        if missing:
            raise ValueError(f'endemic features {missing} not found. Features (in tensor order): {feature_names}')

        # feature positions follow the ColumnRegistry order = feature axis of x
        case_idx    = feature_names.index(case_feature)
        season_idx  = [feature_names.index(f) for f in endemic_features]

        self.model = HHH4GNNModule(
            num_nodes     = len(self.databuilder.dataorchestrator.data_context.local_shapedata),
            seq_length    = self.epiconfig.sequence_length,
            horizon_size  = self.epiconfig.horizon_size,
            case_idx      = case_idx,
            season_idx    = season_idx,
            mean_count    = self._train_target_mean(),
            node_penalty  = node_penalty,
        ).to(self.device)

        self.config_info['model_hparams'] = {
            'node_penalty':     node_penalty,
            'case_feature':     case_feature,
            'endemic_features': list(endemic_features),
        }

        self._update_status('model_hparams_set')

    # ======================================================================= #
    # explanation
    # ======================================================================= #
    def forecast_components(self, dataset: DataSetSplit = 'test') -> pd.DataFrame:
        """
        The three branches of the NB mean for every (t0, node, horizon), and
        their shares of mu. ``target_time`` is the forecast week, ``target``
        the observed count there.
        """
        self._check_status(['model_hparams_set', 'trained'])
        self.model.eval()

        comps = {k: [] for k in self.model.component_names}
        mus, ys = [], []
        with torch.no_grad():
            for snapshot in self._get_dataloader(dataset):
                snapshot = snapshot.to(self.device)
                (mu, _), parts = self.model(snapshot.x,
                                            snapshot.graph.edge_index,
                                            snapshot.graph.edge_weight,
                                            return_components=True)
                for k in comps:
                    comps[k].append(parts[k].cpu().numpy())
                mus.append(mu.cpu().numpy())
                ys.append(snapshot.y.cpu().numpy())

        mu = np.stack(mus)                                   # [T, N, H]
        y  = np.stack(ys)
        T, N, H = mu.shape
        t0 = pd.to_datetime(self._t0_timestamps(dataset, T))

        frames = []
        for hh in range(H):
            steps = self.epiconfig.horizon_leadtime + hh
            df = pd.DataFrame({
                self.epiconfig.temporal_column: np.repeat(t0, N),
                'target_time':                  np.repeat(t0 + pd.Timedelta(weeks=steps), N),
                self.epiconfig.id_column:       np.tile(np.arange(N), T),
                'horizon':                      hh,
            })
            for k, v in comps.items():
                df[k] = np.stack(v)[:, :, hh].reshape(-1)
            df['mu']     = mu[:, :, hh].reshape(-1)
            df['target'] = y[:, :, hh].reshape(-1)
            for k in comps:
                df[f'share_{k}'] = df[k] / df['mu']
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    def component_table(self, dataset: DataSetSplit = 'test', horizon: int = 0) -> pd.DataFrame:
        """
        Per node: summed branches over the split, their shares of mu (mu-weighted,
        so busy weeks count more), and the ratio of predicted to observed cases.
        """
        comp = self.forecast_components(dataset)
        comp = comp[comp['horizon'] == horizon]
        id_col = self.epiconfig.id_column
        parts = list(self.model.component_names)

        tab = comp.groupby(id_col)[parts + ['mu', 'target']].sum().reset_index()
        for k in parts:
            tab[f'share_{k}'] = tab[k] / tab['mu']
        tab['pred_obs_ratio'] = tab['mu'] / tab['target'].where(tab['target'] > 0)

        names = self.context_data.nodenames[[id_col, f'{self.epiconfig.level}_name']]
        return names.merge(tab, on=id_col, how='right')

    def show_decomposition(self,
                           node_idx: int | str | list[int | str] = 0,
                           dataset:  DataSetSplit = 'test',
                           horizon:  int = 0) -> tuple[Figure, list[Axes]]:
        """
        Stacked expected cases per branch over time, with the observed counts,
        for one or more nodes (``'national'`` = sum over all nodes).

        Parameters
        ----------
        node_idx : int | 'national' | list
            Node(s) to show, one panel each.
        dataset : DataSetSplit
            Split to show (``'train'``, ``'val'`` or ``'test'``).
        horizon : int
            Horizon index (0 = the configured lead time).
        """
        comp  = self.forecast_components(dataset)
        comp  = comp[comp['horizon'] == horizon]
        nodes = node_idx if isinstance(node_idx, list) else [node_idx]
        parts = list(self.model.component_names)
        id_col = self.epiconfig.id_column

        fig, axes_array = plt.subplots(len(nodes), 1, figsize=(16, 2 + 4.5 * len(nodes)), squeeze=False)
        axes: list[Axes] = list(axes_array.flatten())

        for ax, node in zip(axes, nodes):
            if node == 'national':
                df    = comp.groupby('target_time')[parts + ['target']].sum().reset_index()
                title = 'nationally aggregated'
            elif isinstance(node, int):
                if not 0 <= node < self.context_data.num_nodes:
                    raise ValueError(f'node {node} invalid: there are {self.context_data.num_nodes} nodes')
                df    = comp[comp[id_col] == node].sort_values('target_time')
                names = self.context_data.nodenames
                name  = names.loc[names[id_col] == node, f'{self.epiconfig.level}_name'].iloc[0]
                title = f'{name} [{id_col} {node}]'
            else:
                raise ValueError(f"node_idx must be an int or 'national', got {node!r}")

            ax.stackplot(df['target_time'], *[df[k] for k in parts],
                         labels=parts, colors=[COMPONENT_COLORS[k] for k in parts], alpha=0.85)
            ax.plot(df['target_time'], df['target'], color='black', marker='o', markersize=3,
                    linewidth=1.2, label='observed')

            totals = df[parts].sum()
            shares = ', '.join(f'{k} {totals[k] / totals.sum():.0%}' for k in parts)
            ax.set_title(f'{title}   ({shares})')
            ax.set_ylabel('expected cases')
            ax.legend(loc='upper left')
            ax.grid(alpha=0.3)

        steps = self.epiconfig.horizon_leadtime + horizon
        suptitle = f'decomposition of the expected cases by {self.name}, {steps}{self.epiconfig.temporal_frequency} ahead'
        if dataset != 'test':
            suptitle += f' [{dataset}]'
        fig.suptitle(suptitle, fontweight='bold', fontsize=14)
        fig.tight_layout()
        plt.close()
        return fig, axes

    # ======================================================================= #
    # helpers
    # ======================================================================= #
    def _train_target_mean(self) -> float:
        """mean count of the training targets; starts each branch at a third of it"""
        ys = torch.stack([s.y for s in self.databuilder.dataloader_train])
        return float(ys.mean())

    def _check_counts(self) -> None:
        if self.epiconfig.target_column != 'cases' or self.epiconfig.lag_column != 'cases':
            raise ValueError("HHH4GNNModel needs raw counts: EpiConfig(target_column='cases', lag_column='cases'), "
                             f"got target_column={self.epiconfig.target_column!r}, "
                             f"lag_column={self.epiconfig.lag_column!r}.")