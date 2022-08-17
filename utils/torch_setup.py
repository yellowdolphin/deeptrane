import sys
from pathlib import Path


def torchmetrics_fork():
    "Check if torchmetrics is installed and if it is xla-compatible"
    install_dirs = [Path(p) / 'torchmetrics' for p in sys.path if (Path(p) / 'torchmetrics').exists()]
    if len(install_dirs) == 1:
        about_file = install_dirs[0] / '__about__.py'
        metric_module_file = install_dirs[0] / 'metrics.py'

        if metric_module_file.exists():
            with open(metric_module_file, 'r') as f:
                code = f.read()

            if 'if dist_sync_fn is None' in code:
                print("[ √ ] torchmetrics installation: dist_sync_fn issue is fixed")
                return 'xla_fork'

        if about_file.exists():
            with open(about_file, 'r') as f:
                code = f.readlines()
            for l in lines:
                if l.startswith('__version__ ='):
                    version = l.split('=')[1].strip(' \'"\n')
                    return version

    return 'orig' if install_dirs else None
