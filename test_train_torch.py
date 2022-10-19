import os
from config import Config
from utils.general import quietly_run
import json


def test_default():
    cfg = Config('configs/defaults')
    quietly_run('python train.py', debug=True)
    assert 'out_dir' in cfg
    out_dir = cfg.out_dir
    use_folds = cfg.use_folds
    for fold in use_folds:
        metrics_file = f'{out_dir}/metrics_fold{fold}.json'
        pth_file = f'{out_dir}/defaults_fold{fold}_ep1.pth'
        opt_file = f'{out_dir}/defaults_fold{fold}_ep1.opt'
        sched_file = f'{out_dir}/defaults_fold{fold}_ep1.sched'
        assert os.path.exists(metrics_file), f'no metrics file {metrics_file}'
        assert os.path.exists(pth_file), f'no model saved: {pth_file} missing'
        assert os.path.exists(opt_file), f'no optimizer state saved: {opt_file} missing'
        assert os.path.exists(sched_file), f'no scheduler state saved: {sched_file} missing'

        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        assert 'train_loss' in metrics, f'metrics {metrics_file} has no train_loss'
        assert '1' in metrics['train_loss'], f'metrics {metrics_file} train_loss has no epoch 1'
        train_loss = metrics['train_loss']['1']
        assert train_loss < 10, f'large train loss: {train_loss}'


