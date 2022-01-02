import os
import sys
from pathlib import Path
import argparse
import importlib

import yaml

def save_yaml(filepath, content, width=120):
    with open(filepath, "w") as f:
        yaml.dump(content, f, width=width)


def load_yaml(filepath):
    with open(filepath, "r") as f:
        content = yaml.safe_load(f)
    return content


class DotDict(dict):
    """dot.notation access to dictionary attributes
    Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    """  # NOQA

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config_file", default='configs/study_nfnet_v001', help="config file path")
parser.add_argument("-M", "--mode", default='train', help="mode type")
#parser.add_argument("-S", "--stage", default=0, help="stage")
#parser.add_argument("-W", "--wei_dir", default='.', help="test weight dir")

parser_args, _ = parser.parse_known_args(sys.argv)

print("[ √ ] Config file:", parser_args.config_file)
print("[ √ ] Mode:       ", parser_args.mode)

config_file = parser_args.config_file
if config_file.endswith('.yaml'):
    cfg = load_yaml(config_file)
else:
    sys.path.append(str(Path(config_file).parent))
    cfg = importlib.import_module(Path(config_file).stem).cfg

out_dir = Path(cfg["out_dir"])
os.makedirs(out_dir, exist_ok=True)
save_yaml(out_dir / "cfg.yaml", cfg)

cfg = DotDict(cfg)
mixed_precision = cfg.mixed_precision

# Experiment config file name convention:  <tag>_v001.py  ->  <tag>_v001_fold0.pth
if "name" not in cfg:
    cfg.name = Path(__file__).stem
if "tag" not in cfg:
    cfg.tag = cfg.name.split("_v")[0]

# Data Partitioning
if "train_on_all" not in cfg:
    cfg.train_on_all = False
if "num_folds" not in cfg:
    cfg.num_folds = 50 if cfg.tag.startswith('pretrain') else 5
if "use_folds" not in cfg:
    cfg.use_folds = [0]

# Image params
if "size" not in cfg: 
    cfg.size = (384, 384)

# Dataloader
if "multilabel" not in cfg:
    cfg.multilabel = False
if "use_albumentations" not in cfg:
    cfg.use_albumentations = True
if "num_tpu_cores" not in cfg:
    cfg.num_tpu_cores = 1
if "bs" not in cfg:
    cfg.bs = 8
if "epochs" not in cfg:
    cfg.epochs = 1
if "batch_verbose" not in cfg:
    cfg.batch_verbose = 20 // cfg.num_tpu_cores
if "do_class_sampling" not in cfg:
    cfg.do_class_sampling = False
if "frac" not in cfg:
    cfg.frac = 1
if "use_batch_tfms" not in cfg:
    cfg.use_batch_tfms = False

# Optimizer, Scheduler
if "n_acc" not in cfg:
    cfg.n_acc = 1
if "lr_head" not in cfg:
    cfg.lr_head = 1e-4
if "lr_bn" not in cfg:
    cfg.lr_bn = cfg.lr_head
if "lr_body" not in cfg:
    cfg.lr_body = cfg.lr_head
if "betas" not in cfg:
    cfg.betas = (0.9, 0.999)
if "one_cycle" not in cfg:
    cfg.one_cycle = True
if "div_factor" not in cfg:
    cfg.div_factor = 1                            # default: 25, from Chest14: 1
if "pct_start" not in cfg:
    cfg.pct_start = 0.3                           # default: 0.3, from Chest14: 0.6, pipeline1: 0
if "reduce_on_plateau" not in cfg:
    cfg.reduce_on_plateau = False                 # else use Step/MultiStep
if "step_lr_epochs" not in cfg:
    cfg.step_lr_epochs = None                     # scheduler.step() does not work properly
if "step_lr_gamma" not in cfg:
    cfg.step_lr_gamma = 1

# Model
if "use_timm" not in cfg:
    cfg.use_timm = True           # timm or torchvision pretrained models
if "arch_name" not in cfg:
    cfg.arch_name = (
        'cait_xs24_384' if 'cait' in cfg.tag else
        'eca_nfnet_l1' if 'nfnet' in cfg.tag else
        'tf_efficientnetv2_s_in21ft1k')
if "use_aux_loss" not in cfg:
    cfg.use_aux_loss = False
if "seg_weight" not in cfg:
    cfg.seg_weight = 0.4
if "add_hidden_layer" not in cfg:
    cfg.add_hidden_layer = False   # add hidden_layer from efficientnet to smp model
if "use_gem" not in cfg:
    cfg.use_gem = False            # GeM pooling
if "dropout_ps" not in cfg:
    cfg.dropout_ps = [0.5]
if "lin_ftrs" not in cfg:
    cfg.lin_ftrs = []
if "bn_eps" not in cfg:
    cfg.bn_eps = 1e-3              # torch default 1e-5
if "bn_momentum" not in cfg:
    cfg.bn_momentum = 0.1
if "wd" not in cfg:
    cfg.wd = 0.05                  # default 1e-2

if "rst_path" not in cfg:
    cfg.rst_path = None
if "rst_name" not in cfg:
    cfg.rst_name = None
if "reset_opt" not in cfg:
    cfg.reset_opt = False          # don't load optimizer/scheduler state dicts

if "xla" not in cfg:
    cfg.xla = False

if "ema" not in cfg:
    cfg.ema = False
if "muliscale" not in cfg:
    cfg.muliscale = False
if "tta" not in cfg:
    cfg.tta = False

cfg.mode = parser_args.mode
