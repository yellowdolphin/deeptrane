from subprocess import run

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