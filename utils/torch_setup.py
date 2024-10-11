import os
import sys
from pathlib import Path


# silence albumentations propaganda
os.environ['NO_ALBUMENTATIONS_UPDATE'] = "1"


def torchmetrics_version():
    "Check if torchmetrics is installed and if it is xla-compatible"
    install_dirs = [Path(p) / 'torchmetrics' for p in sys.path if (Path(p) / 'torchmetrics').exists()]
    if len(install_dirs) == 1:
        about_file = install_dirs[0] / '__about__.py'
        metric_module_file = install_dirs[0] / 'metrics.py'

        if metric_module_file.exists():
            with open(metric_module_file, 'r') as f:
                code = f.read()

            if 'self.distributed_available_fn = kwargs.pop("distributed_available_fn"' in code:
                # dist_sync_fn issue (#1301) has been fixed in 5bbad47ef7e044d23fe06ab179d3c06b444d0f7e
                print("[ √ ] torchmetrics is xla-compatible")
                return 'xla_compatible'

        if about_file.exists():
            with open(about_file, 'r') as f:
                code = f.readlines()
            for l in code:
                if l.startswith('__version__ ='):
                    version = l.split('=')[1].strip(' \'"\n')
                    return version

    return 'orig' if install_dirs else None
