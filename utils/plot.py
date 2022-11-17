from pathlib import Path
from glob import glob
from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.display import display


def plot_metrics(metrics_files=None, prefix="metrics_fold"):
    """Plot metrics from PyTorch training (json) or Tensorflow logs (csv).

    If no `metrics_files` are passed, search for files starting with `prefix`."""

    best_ep_metrics = []
    metrics_files = sorted(glob(f'{prefix}*.json'))
    tf_metrics_files = sorted(glob(f'{prefix}*.csv'))

    # PyTorch metrics
    for fn in metrics_files:
        print(f"Metrics from {fn}")
        df = pd.read_json(fn)
        display(df)

        title = '_'.join(Path(fn).stem.split('_')[1:])
        losses = df.iloc[:, :2]
        obj_cols = [c for c in df.columns if df[c].dtype == 'O']
        metrics = df.iloc[:, 2:].drop(columns=['lr', 'Wall'] + obj_cols, errors='ignore')
        losses_metrics = pd.concat([losses, metrics])
        best_metric = df.iloc[:, -3]
        lr = df.loc[:, ['lr' if 'lr' in df else 'lr_head']]
        best_ep = best_metric.index[best_metric.argmax()]

        losses.plot(title=title)
        metrics.plot(title=title)
        lr.plot(title=title)

        # Print metrics for spreadsheet copy&paste (tab-separated)
        keys = chain(['best_ep'], losses.columns.tolist(), metrics.columns.tolist(),
                     [f'avg_{best_metric.name}', 'Wall(min/ep)'])
        values = chain([str(best_ep)],
                       [f'{df.loc[best_ep, c]:.5f}' for c in losses_metrics.columns],
                       [f'{best_metric.mean():.5f}', f'{df.Wall.mean():.2f}'])
        print('\t'.join(keys))
        print('\t'.join(values))
        best_ep_metrics.append(df.loc[[best_ep]])

    if len(metrics_files) > 1:
        best_ep_metrics = pd.concat(best_ep_metrics)
        best_ep_metrics.index.name = f'best_{best_metric.name}_ep'
        best_ep_metrics.reset_index(inplace=True)
        best_ep_metrics = pd.concat([best_ep_metrics, best_ep_metrics.mean().to_frame('mean').T])
        display(best_ep_metrics)

    # Tensorflow metrics
    for csv in tf_metrics_files:
        df = pd.read_csv(csv, index_col='epoch')
        df.index += 1

        losses = [c for c in df.columns if 'loss' in c]
        # CSVLogger does not write columns in order of metrics (maybe inverse order?)
        metrics = [c for c in df.columns[::-1] if c.startswith('val_') and c not in losses]

        best_metric = metrics[-1]
        df[losses].plot()
        df[metrics].plot()
        plt.show()
        df.lr.plot(title='lr')

        # Print metrics for spreadsheet copy&paste (tab-separated)
        best_idx = df[best_metric].argmax()
        best_ep = df.index[best_idx]
        fields = losses + metrics
        best_metrics = df.loc[best_ep, fields].tolist()
        keys = ['best_ep'] + fields
        values = [str(best_ep)] + [f'{m:.5f}' for m in best_metrics]
        print('\t'.join(keys))
        print('\t'.join(values))
