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
