from pathlib import Path
from glob import glob
from itertools import chain
import matplotlib.pyplot as plt
import pandas as pd
import torch
from IPython.core.display import display


def plot_metrics(metrics_files=None, prefix="metrics_fold"):
    """Plot metrics from PyTorch training (json) or Tensorflow logs (csv).

    If no `metrics_files` are passed, search for files starting with `prefix`."""

    best_ep_metrics = []
    if metrics_files:
        torch_metrics_files = sorted(fn for fn in metrics_files if fn.lower().endswith('.pth'))
        tf_metrics_files = sorted(fn for fn in metrics_files if fn.lower().endswith('.csv'))
    else:
        torch_metrics_files = sorted(glob(f'{prefix}*.pth'))
        tf_metrics_files = sorted(glob(f'{prefix}*.csv'))

    # PyTorch metrics
    for fn in torch_metrics_files:
        print(f"Metrics from {fn}")
        metrics_dict = torch.load(fn)
        df = pd.DataFrame(metrics_dict).set_index('epoch')
        display(df)

        title = '_'.join(Path(fn).stem.split('_')[1:])
        loss_cols = ['train_loss', 'valid_loss']
        losses = df.loc[:, loss_cols]
        obj_cols = [c for c in df.columns if df[c].dtype == 'O']
        assert not obj_cols, f'dtype "O": {obj_cols}'
        #metrics = df.iloc[:, 2:].drop(columns=['lr', 'Wall'] + obj_cols, errors='ignore')
        metrics = df.drop(columns=['lr', 'Wall'] + loss_cols)
        losses_metrics = pd.concat([losses, metrics])
        best_metric = losses_metrics.iloc[:, -1]  # assume last metric is most important
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

    if len(torch_metrics_files) > 1:
        best_ep_metrics = pd.concat(best_ep_metrics)
        best_ep_metrics.index.name = f'best_{best_metric.name}_ep'
        best_ep_metrics.reset_index(inplace=True)
        best_ep_metrics = pd.concat([best_ep_metrics, best_ep_metrics.mean().to_frame('mean').T])
        display(best_ep_metrics)

    if torch_metrics_files: return

    # Tensorflow metrics
    for csv in tf_metrics_files:
        df = pd.read_csv(csv, index_col='epoch')

        losses = [c for c in df.columns if 'loss' in c]
        # CSVLogger does not write columns in order of metrics (maybe inverse order?)
        metrics = [c for c in df.columns[::-1] if c.startswith('val_') and c not in losses]
        if not metrics:
            # this is actually a pytorch metrics csv file
            metrics = [c for c in df.columns if c not in ['lr', 'Wall'] + losses]

        epoch_range = df.index.min(), df.index.max()
        df[losses].plot(xlim=epoch_range)
        df[metrics].plot(xlim=epoch_range)
        plt.show()

        # Print metrics for spreadsheet copy&paste (tab-separated)
        best_metric = metrics[-1]
        best_idx = df[best_metric].argmax()
        best_ep = df.index[best_idx]
        fields = losses + metrics
        best_metrics = df.loc[best_ep, fields].tolist()
        keys = ['best_ep'] + fields
        values = [str(best_ep)] + [f'{m:.5f}' for m in best_metrics]
        print('\t'.join(keys))
        print('\t'.join(values))

        df.lr.plot(title='lr', xlim=epoch_range)
