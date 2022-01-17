from subprocess import run
from typing import Iterable

def quietly_run(*commands, debug=False):
    "Run `commands` in subprocess, only report on errors, unless debug is True"
    for cmd in commands:
        if debug:
            print(f'$ {cmd}')
            run(cmd.split())
            continue

        res = run(cmd.split(), capture_output=True)
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
    if not key in cfg:
        raise KeyError(f"Unrecognized key: `{key}` is not an attribute of cfg")
    if isinstance(cfg[key], bool):
        cfg[key] = value.lower() == 'true'
    elif isinstance(cfg[key], int) or key in none_or_int:
        cfg[key] = int(value)
    elif isinstance(cfg[key], float):
        cfg[key] = float(value)
    elif value.lower() == 'none':
        cfg[key] = [] if isinstance(cfg[key], list) else None
    elif isinstance(cfg[key], str) or cfg[key] is None:
        cfg[key] = str(value)
    else:
        raise TypeError(f"Type {type(cfg[key])} of `{key}` not supported by `--set`")
    return None
