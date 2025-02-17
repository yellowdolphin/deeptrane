import os
import gc
import sys
from glob import glob
from pathlib import Path
import importlib
from multiprocessing import cpu_count
import types
import warnings
#warnings.filterwarnings('ignore', category=FutureWarning, message=re.escape('weights_only')) # no effect
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

from config import Config, parser, DotDict
from utils.general import quietly_run, listify, sizify, autotype, get_drive_out_dir

from metadata import get_metadata
from models import get_pretrained_timm2, get_smp_model

# Import project (code, constant settings)
project = importlib.import_module('projects.autolevels')


DEBUG = False
cloud = 'kaggle' if 'KAGGLE_DOCKER_IMAGE' in os.environ else 'drive' if os.path.exists('/content') else 'gcp'
use_timm = True

print("[ √ ] Cloud:", cloud)
if cloud == 'kaggle':
    print("      Docker Image:", os.environ.get('KAGGLE_DOCKER_IMAGE', '?'))
print(f"[ √ ] {cpu_count()} CPUs")

# Install torch.xla on TPU supported nodes
tpu_vars = 'TPU_ACCELERATOR_TYPE TPU_PROCESS_ADDRESSES PYTORCH_LIBTPU PIP_LIBTPU ACCELERATOR_TYPE AGENT_BOOTSTRAP_IMAGE TPU_SKIP_MDS_QUERY TPU_TOPOLOGY_WRAP TPU_HOST_BOUNDS'.split()
tpu_vars.extend(['COLAB_TPU_ADDR', 'XRT_TPU_CONFIG'])  # colab
found_xla = any([v in os.environ for v in tpu_vars])
if found_xla:
    # This is necessary to make xmp.spawn(process, start_method='fork') work:
    #print(f"TPU_PROCESS_ADDRESSES: {os.environ.get('TPU_PROCESS_ADDRESSES', None)}")  # local
    if 'TPU_PROCESS_ADDRESSES' in os.environ:
        os.environ.pop('TPU_PROCESS_ADDRESSES')  # see https://github.com/pytorch/xla/issues/8215

    xla_nightly = False
    # '1.8.1' works on kaggle and colab, nightly only on kaggle
    #xla_version, apt_libs = ('nightly', '--apt-packages libomp5 libopenblas-dev') if xla_nightly else ('1.8.1', '')
    xla_version, apt_libs = ('nightly', '--apt-packages libomp5 libopenblas-dev') if xla_nightly else ('1.10.0', '')
    # Auto installation
    if (cloud == 'drive'):
        # Colab runs now python 3.10
        # check xla_version for python 3.10: $ gsutil ls gs://tpu-pytorch/wheels/colab/*cp310*.whl
        # There is only one python 3.10 wheel for torch_xla, none for torch/torchvision.
        # Normal pip install works for torch/torchvision, though.
        # Also cloud-tpu-client must be installed.
        try:
            import torch_xla
        except ModuleNotFoundError:
            wheel = 'https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-2.0-cp310-cp310-linux_x86_64.whl'
            quietly_run(f'pip install cloud-tpu-client==0.10 torch==2.0.0 torchvision==0.15.1 {wheel}', debug=False)
            import torch_xla
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
                debug=DEBUG)
            print("LD_LIBRARY_PATH:", os.environ['LD_LIBRARY_PATH'])
    elif not os.path.exists('/opt/conda/lib/python3.7/site-packages/torch_xla'):
        try:
            import torch_xla #_NO_DONT_IMPORT_TORCH2_XLA
            xla_version = torch_xla.__version__
        except ModuleNotFoundError:
            print("running pytorch-xla-env-setup.py --version 1.10.0 ...")
            quietly_run(
                f'{sys.executable} pytorch-xla-env-setup.py --version 1.10.0',
                debug=DEBUG)
            # for some reason does not install torch
            #quietly_run('pip install torch==1.10.0', debug=True)
    print("[ √ ] Python:", sys.version.replace('\n', ''))
    print("[ √ ] XLA:", xla_version, f"(XLA_USE_BF16: {os.environ.get('XLA_USE_BF16', None)})")
    # Install catalyst, required by DistributedSamplerWrapper
    try:
        import catalyst
    except ModuleNotFoundError:
        #quietly_run('pip install -U --progress-bar off catalyst', debug=True)  # fails due to "torch>=1.4" is false
        quietly_run('pip install --no-deps --progress-bar off catalyst', debug=False)
        import catalyst
    print("[ √ ] catalyst:", catalyst.__version__)
import torch
print("[ √ ] torch:", torch.__version__)

# Install torchmetrics
quietly_run('pip install torchmetrics>=0.11.1')

# Install timm
if use_timm:
    try:
        import timm
    except ModuleNotFoundError:
        wheels_path = '/kaggle/input/popular-wheels' if cloud == 'kaggle' else None
        pip_option = f'-f file://{wheels_path}' if wheels_path else ''
        quietly_run(f'pip install {pip_option} timm', debug=DEBUG)
        import timm
    print("[ √ ] timm:", timm.__version__)


from xla_train import _mp_fn

def launch_mp_fns(rank, configs, metadatas, models):
    xm.master_print("lauching jobs...")
    cfg = DotDict(configs[rank])
    model = models[rank]
    metadata = metadatas[rank]
    return _mp_fn(rank, cfg, metadata, model, xm, 0)


# Read config files and parser_args
parser_args, _ = parser.parse_known_args(sys.argv)
if found_xla:
    print("configs:", parser_args.config_files)
    assert len(parser_args.config_files) == 8, f'need 8 configs for 8 TPU cores, got {len(parser_args.config_files)}!'

configs = []
metadatas = []
models = []
for job_id, config_file in enumerate(parser_args.config_files):
    print(f"\n    --- job {job_id} ---")
    cfg = Config('configs/defaults')
    cfg.update(config_file)

    cfg.DEBUG = DEBUG
    cfg.mode = parser_args.mode
    cfg.use_folds = parser_args.use_folds or cfg.use_folds
    cfg.epochs = parser_args.epochs or cfg.epochs
    cfg.batch_verbose = parser_args.batch_verbose or cfg.batch_verbose
    cfg.size = cfg.size if parser_args.size is None else sizify(parser_args.size)
    cfg.metrics = parser_args.metrics or cfg.metrics
    cfg.betas = parser_args.betas or cfg.betas
    for key in 'dropout_ps lin_ftrs freeze'.split():
        setattr(cfg, key, cfg[key] if getattr(parser_args, key) is None else listify(getattr(parser_args, key)).copy())
    for key, value in listify(parser_args.set):
        autotype(cfg, key, value)
    print(f"[ √ ] lin_ftrs from {config_file}/parser: {cfg.lin_ftrs}")

    cfg.cloud = cloud
    if cfg.cloud == 'drive':
        cfg.out_dir = get_drive_out_dir(cfg)  # config.yaml and experiments go there

    print(cfg)
    print("[ √ ] Tags:", cfg.tags)
    print("[ √ ] Mode:", cfg.mode)
    print("[ √ ] Folds:", cfg.use_folds)
    print("[ √ ] Architecture:", cfg.arch_name)

    out_dir = Path(cfg.out_dir) / f'job_{job_id}'
    os.makedirs(out_dir, exist_ok=True)
    cfg.save_yaml(out_dir / 'config.yaml')
    cfg.out_dir = out_dir

    # Config consistency checks
    if cfg.rst_name is not None:
        rst_file = Path(cfg.rst_path) / f'{cfg.rst_name}.pth'
        assert rst_file.exists(), f'{rst_file} not found'  # fail early

    project.init(cfg)

    metadata = get_metadata(cfg, project)
    metadata.to_json(cfg.out_dir / 'metadata.json')

    use_fold = cfg.use_folds[0]
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
        pretrained_model = get_pretrained_timm2(cfg)
    pretrained_model.requires_labels = getattr(pretrained_model, 'requires_labels', False)
    #print(pretrained_model)
    if hasattr(pretrained_model, 'head'):
        print(pretrained_model.head)
    if hasattr(pretrained_model, 'model') and hasattr(pretrained_model.model, 'head'):
        print(pretrained_model.model.head)
    if hasattr(pretrained_model, 'arc'):
        print(pretrained_model.arc)

    # Print parameter counts
    num_params, num_trainable, num_el, num_trainable_el = 0, 0, 0, 0
    for p in pretrained_model.parameters():
        num_params += 1
        num_el += p.numel()
        if p.requires_grad: 
            num_trainable += 1
            num_trainable_el += p.numel()
    print("="*50)
    print(f"Total params:         {num_params:6d} {num_el:20,}")
    print(f"Trainable params:     {num_trainable:6d} {num_trainable_el:20,}")
    print(f"Non-trainable params: {num_params - num_trainable:6d} {num_el - num_trainable_el:20,}")
    print()

    if cfg.compile_torch_model and hasattr(torch, 'compile'):
        from time import perf_counter
        print("compiling model...")
        t0 = perf_counter()
        pretrained_model = torch.compile(pretrained_model)
        print(f"model compiled in {perf_counter() - t0:.1f} sec.")

    cfg.xla = found_xla

    # Drop cfg items that cannot be pickled, they can't be passed to _mp_fn.
    pickleable_cfg = {key: value for key, value in cfg.items() 
                      if not isinstance(value, (types.FunctionType, types.MethodType))}
    for key in cfg.keys():
        if key not in pickleable_cfg:
            print(f"excluding function/method {key} from cfg passed to _mp_fn")

    configs.append(pickleable_cfg)
    metadatas.append(metadata)
    models.append(pretrained_model)


if found_xla:
    # Start cfg/job-distributed training on TPU cores
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.debug.metrics as met

    print("calling xmp.spawn(start_method='fork')...")
    xmp.spawn(launch_mp_fns, start_method='fork', args=(configs, metadatas, models))

    if cfg.xla_metrics:
        xm.master_print()
        report = met.metrics_report()  # str
        if 'XrtTryFreeMemory' in report:
            xm.master_print("XrtTryFreeMemory: reduce bs!")
        xm.master_print(report)

else:
    # Train on CPU/GPU if no xla

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
        def get_ordinal():
            return 0

        @staticmethod
        def save(*args, **kwargs):
            torch.save(*args, **kwargs)

        @staticmethod
        def mesh_reduce(tag, data, reduce_fn):
            return reduce_fn([data])


    for cfg, metadata, model in zip(configs, metadatas, models):
        local_rank = int(os.environ['LOCAL_RANK']) if cfg.get('use_ddp', False) else None
        use_fold = cfg['use_folds'][0]
        _mp_fn(local_rank, DotDict(cfg), metadata, model, xm, use_fold)
        gc.collect()
        torch.cuda.empty_cache()
