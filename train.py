import os
import gc
import sys
from glob import glob
from pathlib import Path
import importlib
from multiprocessing import cpu_count
#import warnings
#warnings.filterwarnings('ignore')

from config import Config, parser
from utils.general import quietly_run, listify, sizify, autotype, get_drive_out_dir
from utils.torch_setup import torchmetrics_version

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
cfg.metrics = parser_args.metrics or cfg.metrics
cfg.betas = parser_args.betas or cfg.betas
cfg.dropout_ps = cfg.dropout_ps if parser_args.dropout_ps is None else listify(parser_args.dropout_ps)
cfg.lin_ftrs = cfg.lin_ftrs if parser_args.lin_ftrs is None else listify(parser_args.lin_ftrs)
for key, value in listify(parser_args.set):
    autotype(cfg, key, value)

cfg.cloud = 'drive' if os.path.exists('/content') else 'kaggle' if os.path.exists('/kaggle') else 'gcp'
if cfg.cloud == 'drive':
    cfg.out_dir = get_drive_out_dir(cfg)  # config.yaml and experiments go there

print(cfg)
print("[ √ ] Cloud:", cfg.cloud)
if cfg.cloud == 'kaggle':
    print("      Docker Image:", os.environ.get('KAGGLE_DOCKER_IMAGE', '?'))
print("[ √ ] Tags:", cfg.tags)
print("[ √ ] Mode:", cfg.mode)
print("[ √ ] Folds:", cfg.use_folds)
print("[ √ ] Architecture:", cfg.arch_name)

cfg.save_yaml()

# Config consistency checks
if cfg.frac != 1: assert cfg.do_class_sampling or (cfg.filetype == 'wds'), 'frac w/o class_sampling not implemented'
if cfg.rst_name is not None:
    rst_file = Path(cfg.rst_path) / f'{cfg.rst_name}.pth'
    assert rst_file.exists(), f'{rst_file} not found'  # fail early
cfg.out_dir = Path(cfg.out_dir)

wheels_path = cfg.wheels_path or ('/kaggle/input/popular-wheels' if cfg.cloud == 'kaggle' else None)
pip_option = f'-f file://{wheels_path}' if wheels_path else ''

# Install torch.xla on TPU supported nodes
tpu_vars = 'TPU_ACCELERATOR_TYPE TPU_PROCESS_ADDRESSES PYTORCH_LIBTPU PIP_LIBTPU ACCELERATOR_TYPE AGENT_BOOTSTRAP_IMAGE TPU_SKIP_MDS_QUERY TPU_TOPOLOGY_WRAP TPU_HOST_BOUNDS'.split()
if any([v in os.environ for v in tpu_vars]):
    cfg.xla = True
    # '1.8.1' works on kaggle and colab, nightly only on kaggle
    #xla_version, apt_libs = ('nightly', '--apt-packages libomp5 libopenblas-dev') if cfg.xla_nightly else ('1.8.1', '')
    xla_version, apt_libs = ('nightly', '--apt-packages libomp5 libopenblas-dev') if cfg.xla_nightly else ('1.10.0', '')
    # Auto installation
    if (cfg.cloud == 'drive'):
        # check xla_version for python 3.7: $ gsutil ls gs://tpu-pytorch/wheels/colab | grep cp37
        # 'nightly': installs torch-1.13.0a0+git83c6113, libmkl_intel_lp64.so.1 missing
        # '1.12': on colab re-installs torch etc anyways, libmkl_intel_lp64.so.1 missing
        # '1.11': on colab installs torch-1.11.0a0+git8d365ae, libmkl_intel_lp64.so.1 missing
        # w/o --version, installs torch 1.6.0 which has no MpSerialExecutor
        if False:  # install in notebook instead
            quietly_run(
                'curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py',
                f'{sys.executable} pytorch-xla-env-setup.py --version 1.8.1',
                'pip install -U --progress-bar off catalyst',  # for DistributedSamplerWrapper
                debug=cfg.DEBUG)
        # LD_LIBRARY_PATH points to /usr/local/nvidia, which does not exist
        # xla setup creates /usr/local/lib/libmkl_intel_lp64.so but not .so.1
        print("LD_LIBRARY_PATH:", os.environ['LD_LIBRARY_PATH'])
        # For some reason this does not work:
        #os.environ['LD_LIBRARY_PATH'] += ':/usr/local/lib'  # fix 'libmkl_intel_lp64.so.1 not found'
        #if os.path.exists('/usr/local/lib/libmkl_intel_lp64.so'):
        #    os.symlink('/usr/local/lib/libmkl_intel_lp64.so', '/usr/local/lib/libmkl_intel_lp64.so.1')
    #elif (xla_version != '1.8.1') and not os.path.exists('/opt/conda/lib/python3.7/site-packages/torch_xla/experimental/pjrt.py'):
    elif (xla_version != '1.10.0') and not os.path.exists('/opt/conda/lib/python3.7/site-packages/torch_xla/experimental/pjrt.py'):
        #try:
        #    import torch_xla
        #    xla_version = torch_xla.__version__
        #except ModuleNotFoundError:
        if True:
            print(f"running pytorch-xla-env-setup.py --version {xla_version} {apt_libs} ...")
            quietly_run(
                'curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py',
                f'{sys.executable} pytorch-xla-env-setup.py --version {xla_version} {apt_libs}',
                'pip install -U numpy',  # nightly torch_xla needs newer numpy but does not "require" it
                debug=cfg.DEBUG)
            print("LD_LIBRARY_PATH:", os.environ['LD_LIBRARY_PATH'])
    elif not os.path.exists('/opt/conda/lib/python3.7/site-packages/torch_xla'):
        try:
            import torch_xla
            xla_version = torch_xla.__version__
        except ModuleNotFoundError:
            print("running pytorch-xla-env-setup.py --version 1.10.0 ...")
            quietly_run(
                f'{sys.executable} pytorch-xla-env-setup.py --version 1.10.0',
                debug=cfg.DEBUG)
    print("[ √ ] Python:", sys.version.replace('\n', ''))
    print("[ √ ] XLA:", xla_version, f"(XLA_USE_BF16: {os.environ.get('XLA_USE_BF16', None)})")
    # Install catalyst, required by DistributedSamplerWrapper
    try:
        import catalyst
    except ModuleNotFoundError:
        quietly_run('pip install -U --progress-bar off catalyst')
        import catalyst
    print("[ √ ] catalyst:", catalyst.__version__)
import torch
print("[ √ ] torch:", torch.__version__)

# Install (xla compatible) torchmetrics
#if cfg.xla and torchmetrics_version() != 'xla_compatible':
    # Use own fork while current torchmetrics is broken. FIXED!
    #quietly_run('pip install git+https://github.com/yellowdolphin/metrics.git', debug=cfg.DEBUG)

    # Current version works with torch_xla and torch 1.8.1, but torch requirement is not met by "1.8.0a0+56b43f4",
    # have to prevent re-installation of wrong torch version. FIXED!
    # Switch to pypi when distributed_available fix is out (10.4?)
    #quietly_run('pip install --no-deps git+https://github.com/Lightning-AI/metrics.git', debug=cfg.DEBUG)
    #quietly_run('pip install numpy>=1.17.2 packaging typing-extensions lightning-utilities >=0.7.0, <0.8.0')
#    print("installing current version of torchmetrics with requirements...")
#    quietly_run('pip install -U torchmetrics', debug=True)
#elif not cfg.xla and not torchmetrics_version():
#    quietly_run('pip install torchmetrics>=0.8', debug=cfg.DEBUG)
#else:
#    # require 0.8.0+ for kwarg compute_groups
#    tm_version = torchmetrics_version().split('.')
#    if tm_version[0] == '0' and int(tm_version[1]) < 8:
#        print("installing current version of torchmetrics...")
#        quietly_run('pip install -U torchmetrics', debug=True)

# Install torchmetrics
quietly_run('pip install torchmetrics>=0.11.1')

# Install timm
if cfg.use_timm:
    try:
        import timm
    except ModuleNotFoundError:
        quietly_run(f'pip install {pip_option} timm', debug=cfg.DEBUG)
        import timm
    print("[ √ ] timm:", timm.__version__)

# Install keras if preprocess_inputs is needed
if cfg.cloud == 'kaggle' and cfg.normalize in ['torch', 'tf', 'caffe']:
    quietly_run(f'pip install keras=={tf.keras.__version__}', debug=cfg.DEBUG)
if cfg.filetype == 'wds':
    quietly_run('pip install webdataset', debug=cfg.DEBUG)


print(f"[ √ ] {cpu_count()} CPUs")

# Determine GPU usage
cfg.gpu = not cfg.xla and torch.cuda.is_available()
if not (cfg.gpu and torch.cuda.device_count() > 1):
    cfg.use_dp = cfg.use_ddp = False
else:
    # For now, ddp jobs must be run with torchrun, which sets 'RANK'
    cfg.use_ddp = int(os.environ.get('RANK', -1)) != -1
if cfg.use_ddp:
    cfg.use_dp = False  # ddp overrides dp

# Get distributed namespace for detected accelerator: "xm"
if cfg.xla:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.debug.metrics as met

elif cfg.use_ddp:
    from torch.distributed import ReduceOp

    class xm(object):
        "Pseudo class to overload torch_xla.core.xla_model"
        get_ordinal = torch.distributed.get_rank  # in range(world_size)
        xrt_world_size = torch.distributed.get_world_size

        @classmethod
        def master_print(cls, *args, **kwargs):
            if cls.get_ordinal() == 0: print(*args, **kwargs)

        @classmethod
        def xla_device(cls):
            local_rank = int(os.environ["LOCAL_RANK"])
            global_rank = cls.get_ordinal()
            assert local_rank == global_rank % torch.cuda.device_count()
            return local_rank  # f"cuda:{local_rank}"

        @staticmethod
        def save(*args, **kwargs):
            if torch.distributed.get_rank() == 0:
                torch.save(*args, **kwargs)

        @staticmethod
        def optimizer_step(optimizer, barrier=False):
            optimizer.step()

        @classmethod
        def mesh_reduce(cls, tag, data, reduce_fn):
            "Returns reduced data to process 0 (data to all others)"
            #print(f"mesh_reduce called with tag={tag}, data={data}, reduce_fn={reduce_fn}")
            if reduce_fn.__name__.upper() == 'LIST':
                if isinstance(data, torch.Tensor):
                    data_list = [torch.zeros_like(data) for _ in range(cls.xrt_world_size())]
                    torch.distributed.gather(data, data_list if cls.get_ordinal() == 0 else None, dst=0)
                    return data_list
                else:
                    data_list = [0 for _ in range(cls.xrt_world_size)]
                    torch.distributed.gather_object(data, data_list if cls.get_ordinal() == 0 else None, dst=0)
                    return data_list
            op = getattr(ReduceOp, reduce_fn.__name__.upper())
            device = cls.get_ordinal()
            dtype = torch.float32 if isinstance(data, float) else torch.int64
            data = torch.tensor(data, dtype=dtype, device=device)
            torch.distributed.reduce(data, dst=0, op=op)
            return data  # only rank 0 gets the reduced result

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

# Report accelerators
if cfg.xla:
    cfg.n_replicas = cfg.n_replicas or xm.xrt_world_size()
    print(f"[ √ ] Using {cfg.n_replicas} TPU cores")
elif cfg.gpu:
    cfg.n_replicas = torch.cuda.device_count() if (cfg.use_dp or cfg.use_ddp) else 1
    print(f"[ √ ] Using {cfg.n_replicas} GPUs", "(DP)" if cfg.use_dp else "(DDP)" if cfg.use_ddp else "")
else:
    cfg.n_replicas = 1
    #cfg.bs = min(cfg.bs, 3 * cpu_count())  # avoid RAM exhaustion during CPU debug
    print(f"[ √ ] No accelerators found, reducing bs to {cfg.bs}")

if cfg.use_aux_loss:
    try:
        import segmentation_models_pytorch as smp
    except ModuleNotFoundError:
        quietly_run('pip install -qU git+https://github.com/qubvel/segmentation_models.pytorch')
        import segmentation_models_pytorch as smp
    print("[ √ ] segmentation_models_pytorch:", smp.__version__)

from metadata import get_metadata
from models import get_pretrained_model, get_pretrained_timm, get_smp_model
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
    cfg.NUM_TRAINING_IMAGES = (~ metadata.is_valid).sum()
    cfg.NUM_VALIDATION_IMAGES = metadata.is_valid.sum()
    print(f"Train set: {cfg.NUM_TRAINING_IMAGES:12d}")
    print(f"Valid set: {cfg.NUM_VALIDATION_IMAGES:12d}")

    if hasattr(project, 'pooling'):
        cfg.pooling = project.pooling
    if hasattr(project, 'bottleneck'):
        cfg.bottleneck = project.bottleneck

    if cfg.use_aux_loss:
        pretrained_model = get_smp_model(cfg)
    else:
        pretrained_model = get_pretrained_model(cfg)
        #pretrained_model = get_pretrained_timm(cfg)
    pretrained_model.requires_labels = getattr(pretrained_model, 'requires_labels', False)
    #print(pretrained_model)
    if hasattr(pretrained_model, 'head'):
        print(pretrained_model.head)
    if hasattr(pretrained_model, 'model') and hasattr(pretrained_model.model, 'head'):
        print(pretrained_model.model.head)
    if hasattr(pretrained_model, 'arc'):
        print(pretrained_model.arc)


    fn = cfg.out_dir / f'{cfg.name}_init.pth'
    if False and not fn.exists():
        print(f"Saving initial model as {fn}")
        torch.save(pretrained_model.state_dict(), fn)


    # Start distributed training on TPU cores
    if cfg.xla:
        torch.set_default_tensor_type('torch.FloatTensor')

        # MpModelWrapper wraps a model to minimize host memory usage (fork only)
        pretrained_model = xmp.MpModelWrapper(pretrained_model) if cfg.n_replicas > 1 else pretrained_model

        print("xmp.spawn (fork)")
        xmp.spawn(_mp_fn, nprocs=cfg.n_replicas, start_method='fork',
                  args=(cfg, metadata, pretrained_model, xm, use_fold))

        if cfg.xla_metrics:
            xm.master_print()
            report = met.metrics_report()  # str
            if 'XrtTryFreeMemory' in report:
                xm.master_print("XrtTryFreeMemory: reduce bs!")
            xm.master_print(report)


    # Start distributed training on GPUs
    elif False and cfg.use_ddp:  # does not work, use torchscript instead!
        import torch.multiprocessing as mp

        print(f"mp.spawn {_mp_fn} on {cfg.n_replicas} GPUs with xm={xm}")
        mp.spawn(_mp_fn,
            args=(cfg, metadata, pretrained_model, xm, use_fold),
            nprocs=cfg.n_replicas,
            join=True)


    # Train on CPU/GPU if no xla
    else:
        if cfg.use_dp:
            model_requires_labels = pretrained_model.requires_labels  # stripped by wrapper
            pretrained_model = torch.nn.DataParallel(pretrained_model)
            pretrained_model.requires_labels = model_requires_labels

        _mp_fn(None, cfg, metadata, pretrained_model, xm, use_fold)
        del pretrained_model
        gc.collect()
        torch.cuda.empty_cache()
