import os
from subprocess import run
from typing import Iterable


def quietly_run(*commands, debug=False, cwd=None):
    "Run `commands` in subprocess, only report on errors, unless debug is True."
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


def get_package_version(name):
    """Return version of python package `name` as list (str).
    
    Uses pip to get package version without import so it can be updated if required.
    Requires linux/unix."""
    res = run(f'pip freeze | grep "^{name}=="', shell=True, capture_output=True)
    if res.stdout:
        substrings = res.stdout.decode().split('==')[-1].split('.')
    else:
        res = run(fr'pip freeze | grep "{name} @ .*\.whl"', shell=True, capture_output=True)
        substrings = res.stdout.decode().split('/')[-1].split('-')[1].split('.')
    if len(substrings) < 2:
        print("No (sub)version info from pip freeze for {name}:")
        res = run(f'pip freeze | grep {name}', shell=True, capture_output=True)
        print(res.stdout.decode())
    return substrings


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
    "Update `cfg.key` with `value`, converted from str to inferred type."
    if False:
        print(f"autotype key: {key}, value: {value}, cfg[key]: {cfg[key]}")
    if key not in cfg:
        raise KeyError(f"Unrecognized key: `{key}` is not an attribute of cfg")

    # infer type from value
    if value.lower() == 'true':
        cfg[key] = True
    elif value.lower() == 'false':
        cfg[key] = False
    elif value.lower() == 'none':
        cfg[key] = [] if isinstance(cfg[key], list) else None
    else:
        try:
            cfg[key] = int(value)
        except ValueError:
            try:
                cfg[key] = float(value)
            except ValueError:
                cfg[key] = value  # keep str


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


def get_nested_attr(obj, attr_path):
    """
    Get a nested attribute from an object using a dot-separated string path.
    
    Args:
        obj: The object to get the attribute from
        attr_path (str): Dot-separated path to the attribute (e.g., "a.b.c")
        
    Returns:
        The value of the nested attribute
        
    Raises:
        AttributeError: If any part of the path doesn't exist
    """
    current = obj
    for attr in attr_path.split('.'):
        current = getattr(current, attr)
    return current


def set_nested_attr(obj, attr_path, value):
    """
    Set a nested attribute on an object using a dot-separated string path.
    
    Args:
        obj: The object to set the attribute on
        attr_path (str): Dot-separated path to the attribute (e.g., "a.b.c")
        value: The value to set
        
    Raises:
        AttributeError: If any part of the path except the last doesn't exist
    """
    *path_parts, final_attr = attr_path.split('.')
    current = obj
    
    # Navigate to the parent object
    for attr in path_parts:
        current = getattr(current, attr)
    
    # Set the final attribute
    setattr(current, final_attr, value)
