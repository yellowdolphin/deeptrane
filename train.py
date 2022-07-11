import os
import gc
import sys
from glob import glob
from pathlib import Path
import importlib
from multiprocessing import cpu_count
#import warnings
#warnings.filterwarnings('ignore')

from future import removesuffix
from config import Config, parser
from utils.general import quietly_run, listify, sizify, autotype

# Read config file and parser_args
parser_args, _ = parser.parse_known_args(sys.argv)
print("[ √ ] Config file:", parser_args.config_file)
cfg = Config('configs/defaults')
if parser_args.config_file: cfg.update(parser_args.config_file)

cfg.mode = parser_args.mode
cfg.use_folds = parser_args.use_folds or cfg.use_folds
cfg.epochs = parser_args.epochs or cfg.epochs
cfg.batch_verbose = parser_args.batch_verbose or cfg.batch_verbose
cfg.size = cfg.size if parser_args.size is None else sizify(parser_args.size)
cfg.betas = parser_args.betas or cfg.betas
cfg.dropout_ps = cfg.dropout_ps if parser_args.dropout_ps is None else listify(parser_args.dropout_ps)
cfg.lin_ftrs = cfg.lin_ftrs if parser_args.lin_ftrs is None else listify(parser_args.lin_ftrs)
for key, value in listify(parser_args.set):
    autotype(cfg, key, value)
print(cfg)

print("[ √ ] Tags:", cfg.tags)
print("[ √ ] Mode:", cfg.mode)
print("[ √ ] Folds:", cfg.use_folds)
print("[ √ ] Architecture:", cfg.arch_name)
cfg.save_yaml()

# Config consistency checks
if cfg.frac != 1: assert cfg.do_class_sampling, 'frac w/o class_sampling not implemented'
if cfg.rst_name is not None:
    rst_file = Path(cfg.rst_path) / f'{cfg.rst_name}.pth'
    assert rst_file.exists(), f'{rst_file} not found'  # fail early
if cfg.use_aux_loss: assert cfg.use_albumentations, 'MySiimCovidDataset requires albumentations'
cfg.out_dir = Path(cfg.out_dir)

# Install torch.xla on kaggle TPU supported nodes
if 'TPU_NAME' in os.environ:
    cfg.xla = True         # pull out of cfg?
    xla_version = '1.8.1'  # only '1.8.1' works on python3.7

    # Auto installation
    quietly_run(
        #'curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py',
        f'{sys.executable} pytorch-xla-env-setup.py --version {xla_version}',
        #'pip install -U --progress-bar off catalyst',  # for DistributedSamplerWrapper, catalyst 21.8 already installed
        #debug=True
    )
    print("[ √ ] Python:", sys.version.replace('\n', ''))
    print("[ √ ] XLA:", xla_version)
import torch
print("[ √ ] torch:", torch.__version__)

# Install timm
if cfg.use_timm:
    try:
        import timm
    except ModuleNotFoundError:
        if os.path.exists('/kaggle/input/timm-wheels/timm-0.4.13-py3-none-any.whl'):
            quietly_run('pip install /kaggle/input/timm-wheels/timm-0.4.13-py3-none-any.whl')
        else:
            quietly_run('pip install timm', debug=True)
        import timm
    print("[ √ ] timm:", timm.__version__)

if cfg.xla:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    #import torch_xla.debug.metrics as met
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

        @staticmethod
        def mesh_reduce(tag, data, reduce_fn):
            return reduce_fn([data])

print(f"[ √ ] {cpu_count()} CPUs")
if cfg.xla:
    print(f"[ √ ] Using {cfg.num_tpu_cores} TPU cores")
elif not torch.cuda.is_available():
    cfg.bs = min(cfg.bs, 3 * cpu_count())
    print(f"[ √ ] No GPU found, reducing bs to {cfg.bs}")

if cfg.use_aux_loss:
    try:
        import segmentation_models_pytorch as smp
    except ModuleNotFoundError:
        quietly_run('pip install -qU git+https://github.com/qubvel/segmentation_models.pytorch')
        import segmentation_models_pytorch as smp
    print("[ √ ] segmentation_models_pytorch:", smp.__version__)

from metadata import get_metadata
from models import get_pretrained_model, get_smp_model
from xla_train import _mp_fn

# Import project (code, constant settings)
project = importlib.import_module(f'projects.{cfg.project}') if cfg.project else None
if project:
    print("[ √ ] Project:", cfg.project)
    project.init(cfg)

metadata = get_metadata(cfg, project)
metadata.to_json(cfg.out_dir / 'metadata.json')

for use_fold in cfg.use_folds:
    print(f"\nFold: {use_fold}")
    metadata['is_valid'] = metadata.fold == use_fold
    print(f"Train set: {(~ metadata.is_valid).sum():12d}")
    print(f"Valid set: {   metadata.is_valid.sum():12d}")

    if hasattr(project, 'pooling'):
        cfg.pooling = project.pooling
    if hasattr(project, 'bottleneck'):
        cfg.bottleneck = project.bottleneck

    if cfg.use_aux_loss:
        pretrained_model = get_smp_model(cfg)
    else:
        pretrained_model = get_pretrained_model(cfg)
    pretrained_model.requires_labels = getattr(pretrained_model, 'requires_labels', False)
    if hasattr(pretrained_model, 'head'):
        print(pretrained_model.head)
    if hasattr(pretrained_model, 'model') and hasattr(pretrained_model.model, 'head'):
        print(pretrained_model.model.head)
    if hasattr(pretrained_model, 'arc'):
        print(pretrained_model.arc)

    fn = cfg.out_dir / f'{cfg.name}_init.pth'
    if not fn.exists():
        print(f"Saving initial model as {fn}")
        torch.save(pretrained_model.state_dict(), fn)

    # Start distributed training on TPU cores
    if cfg.xla:
        torch.set_default_tensor_type('torch.FloatTensor')

        # MpModelWrapper wraps a model to minimize host memory usage (fork only)
        wrapped_model = xmp.MpModelWrapper(pretrained_model) if cfg.num_tpu_cores > 1 else pretrained_model
        serial_executor = xmp.MpSerialExecutor()

        xmp.spawn(_mp_fn, nprocs=cfg.num_tpu_cores, start_method='fork',
                  args=(cfg, metadata, wrapped_model, serial_executor, xm, use_fold))

    # Or train on CPU/GPU if no xla
    else:
        wrapped_model = pretrained_model
        _mp_fn(None, cfg, metadata, wrapped_model, None, xm, use_fold)
        del wrapped_model
        del pretrained_model
        gc.collect()
        torch.cuda.empty_cache()

    if cfg.save_best:
        saved_models = sorted(glob(f'{cfg.out_dir / cfg.name}_fold{use_fold}_ep*.pth'))
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
