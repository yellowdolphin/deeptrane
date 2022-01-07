import os
import gc
import sys
from glob import glob
from subprocess import run
from pathlib import Path

from config import Config, parser
from future import removesuffix


# Read config file and parser_args
parser_args, _ = parser.parse_known_args(sys.argv)
print("[ √ ] Config file:", parser_args.config_file)
cfg = Config('configs/defaults')
if parser_args.config_file: cfg.update(parser_args.config_file)

cfg.mode = parser_args.mode
cfg.use_folds = parser_args.use_folds or cfg.use_folds
print("[ √ ] Tags:", cfg.tags)
print("[ √ ] Mode:", cfg.mode)
print("[ √ ] Folds:", cfg.use_folds)
cfg.save_yaml()

# Config consistency checks
if cfg.frac != 1: assert do_class_sampling, 'frac w/o class_sampling not implemented'
if cfg.rst_name is not None:
    rst_file = Path(cfg.rst_path) / f'{cfg.rst_name}.pth'
    assert rst_file.exists(), f'{rst_file} not found'  # fail early

# Install torch.xla on kaggle TPU supported nodes
if (os.path.exists('/opt/conda/lib/python3.7/site-packages/cloud_tpu_client') and not 
    os.path.exists('/opt/conda/lib/python3.7/site-packages/cloud_tpu_profiler')):
    cfg.xla = True        # pull out of cfg?
    xla_version = '1.8.1' # only '1.8.1' works on python3.7

    # Auto installation
    run('curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py'.split(),
        capture_output=True)
    run(['python', 'pytorch-xla-env-setup.py', '--version', xla_version], capture_output=True)
    run('pip install -Uq --progress-bar off catalyst'.split(), capture_output=True)  # for DistributedSamplerWrapper
    print("[ √ ] XLA:", xla_version)

# Install timm
if cfg.use_timm:
    try:
        import timm
    except ModuleNotFoundError:
        run('pip install timm'.split(), capture_output=True)
        import timm
    print("[ √ ] timm:", timm.__version__)

import torch
print("[ √ ] torch:", torch.__version__)
if cfg.xla:
    #import torch_xla
    #import torch_xla.debug.metrics as met
    #import torch_xla.distributed.data_parallel as dp
    #import torch_xla.distributed.parallel_loader as pl
    #import torch_xla.utils.utils as xu
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    #import torch_xla.test.test_utils as test_utils
    #from catalyst.data import DistributedSamplerWrapper
    #from torch.utils.data.distributed import DistributedSampler
else:
    class xm(object):
        "Pseudo class to overload torch_xla.core.xla_model"
        @staticmethod
        def master_print(*args, **kwargs):
            print(*args, **kwargs)
        @staticmethod
        def xla_device():
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        @staticmethod
        def xrt_world_size():
            return 1
        @staticmethod
        def save(*args, **kwargs):
            torch.save(*args, **kwargs)
        @staticmethod
        def optimizer_step(optimizer, barrier=False):
            optimizer.step()
            #print("optimizer.step() skipped")
        @staticmethod
        def mesh_reduce(tag, data, reduce_fn):
            return reduce_fn([data])

#import warnings
#warnings.filterwarnings('ignore')

from multiprocessing import cpu_count
print(f"[ √ ] {cpu_count()} CPUs")
if not cfg.xla and not torch.cuda.is_available():
    cfg.bs = min(cfg.bs, 3 * cpu_count())
    print(f"[ √ ] No GPU found, reducing bs to {cfg.bs}")

if cfg.use_aux_loss:
    try:
        import segmentation_models_pytorch as smp
    except ModuleNotFoundError:
        run('pip install -qU git+https://github.com/qubvel/segmentation_models.pytorch'.split(), capture_output=True)
        import segmentation_models_pytorch as smp
    print("[ √ ] segmentation_models_pytorch:", smp.__version__)

from datasets import get_metadata
metadata = get_metadata(cfg)
metadata.to_json(f'{cfg.out_dir}/metadata.json')

from models import get_pretrained_model, get_smp_model
from xla_train import _mp_fn

for use_fold in cfg.use_folds:
    print(f"Fold: {use_fold}")
    metadata['is_valid'] = metadata.fold == use_fold
    print(f"train_set: {(~ metadata.is_valid).sum():12d}")
    print(f"valid_set: {   metadata.is_valid.sum():12d}")
    
    if cfg.use_aux_loss:
        pretrained_model = get_smp_model(cfg)
    else:
        pretrained_model = get_pretrained_model(cfg)
    
    #print(pretrained_model)

    
    # Start distributed training on TPU cores
    if cfg.xla:
        torch.set_default_tensor_type('torch.FloatTensor')

        # MpModelWrapper wraps a model to minimize host memory usage (fork only)
        wrapped_model = xmp.MpModelWrapper(pretrained_model) if cfg.num_tpu_cores > 1 else pretrained_model
        serial_executor = xmp.MpSerialExecutor()

        xmp.spawn(_mp_fn, nprocs=cfg.num_tpu_cores, start_method='fork', args=(cfg, metadata, wrapped_model, serial_executor, xm, use_fold))

    # Or train on CPU/GPU if no xla
    else:
        wrapped_model = pretrained_model
        _mp_fn(None, cfg, metadata, wrapped_model, xm, use_fold)
        #_mp_fn(None)
        del wrapped_model
        del pretrained_model
        gc.collect()
        torch.cuda.empty_cache()

    if cfg.save_best:
        saved_models = sorted(glob(f'{Path(cfg.out_dir)/cfg.name}_fold{use_fold}_ep*.pth'))
        if len(saved_models) > 1:
            last_saved = removesuffix(saved_models[-1], '.pth')
            print("Best saved model:", last_saved)
            for saved_model in saved_models[:-1]:
                path_stem = removesuffix(saved_model, '.pth')
                for suffix in 'pth opt sched'.split():
                    fn = f'{path_stem}.{suffix}'
                    if os.path.exists(fn): Path(fn).unlink()
        elif len(saved_models) == 1: 
            print(print("Best saved model:", saved_models[0]))
        else:
            print(f"no checkpoints found in {cfg.out_dir}")