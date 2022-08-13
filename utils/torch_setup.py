import sys
from pathlib import Path


def torchmetrics_fork():
    "Check if torchmetrics is installed and if it is xla-compatible"
    install_dirs = [Path(p) / 'torchmetrics' for p in sys.path if (Path(p) / 'torchmetrics').exists()]
    if len(install_dirs) == 1:
        installed = True
        metric_module_file = install_dirs[0] / '__about__.py'

        if metric_module_file.exists():
            with open(metric_module_file, 'r') as f:
                metric_module = f.read()

            if 'if dist_sync_fn is None' in metric_module:
                print("[ √ ] torchmetrics installation: dist_sync_fn issue is fixed")
                return 'xla_fork'

    return 'orig' if installed else None
