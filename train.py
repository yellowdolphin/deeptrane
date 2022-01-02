import os
from subprocess import run
from config import cfg

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

for k, v in cfg.items():
    print(f'{k:25} {v}')