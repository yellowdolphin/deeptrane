import os
from subprocess import run
from typing import Iterable


def quietly_run(*commands, debug=False, cwd=None):
    "Run `commands` in subprocess, only report on errors, unless debug is True"
    for cmd in commands:
        if debug:
            print(f'$ {cmd}')
            run(cmd.split(), cwd=cwd)
            continue

        res = run(cmd.split(), capture_output=True, cwd=cwd)
        if res.returncode:
            print(f'$ {cmd}')
            print(res.stdout.decode())
            raise Exception(res.stderr.decode())


def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, dict): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]


def sizify(o, dims=2):
    if isinstance(o, int): return (o,) * dims
    if isinstance(o, float): return (int(o),) * dims
    if isinstance(o, list) and len(o) == 1: return tuple(o) * dims
    return tuple(o)


def autotype(cfg, key, value):
    "Update `cfg.key` with converted `value`, infer type from `cfg.key`."
    none_or_int = set(['batch_verbose', 'step_lr_after'])
    if key not in cfg:
        raise KeyError(f"Unrecognized key: `{key}` is not an attribute of cfg")
    if isinstance(cfg[key], bool):
        cfg[key] = value.lower() == 'true'
    elif value.lower() == 'none':
        cfg[key] = [] if isinstance(cfg[key], list) else None
    elif isinstance(cfg[key], int) or key in none_or_int:
        cfg[key] = int(value)
    elif isinstance(cfg[key], float):
        cfg[key] = float(value)
    elif isinstance(cfg[key], str):
        cfg[key] = str(value)
    elif cfg[key] is None:
        # infer from value if default is None
        if value.lower() == 'true':
            cfg[key] = True
        elif value.lower() == 'false':
            cfg[key] = False
        try:
            cfg[key] = int(value)
        except ValueError:
            try:
                cfg[key] = float(value)
            except ValueError:
                cfg[key] = value  # keep str
    else:
        raise TypeError(f"Type {type(cfg[key])} of `{key}` not supported by `--set`")
    return None


def get_drive_out_dir(cfg):
    if cfg.project is None:
        return '/content'

    if not os.path.exists('/content/gdrive'):
        from google.colab import drive
        drive.mount('/content/gdrive', force_remount=False)
    project_dir = f'/content/gdrive/MyDrive/{cfg.project}'

    experiment = cfg.experiment or 1

    # create new experiment folder on google drive
    save_dir = f'{project_dir}/runs/{experiment:03d}'
    if experiment > 0 and os.path.exists(save_dir):
        prev_experiment = max(int(f.name) for f in os.scandir(f'{project_dir}/runs'))
        experiment = prev_experiment + 1
        save_dir = f'{project_dir}/runs/{experiment:03d}'
    os.makedirs(save_dir, exist_ok=True)

    return save_dir
