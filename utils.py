from subprocess import run

def quietly_run(*commands)
    "Run `commands` in subprocess, only report on errors"
    for cmd in commands:
        res = run(cmd.split(), capture_output=True)
        if res.returncode:
            print(f'$ {cmd}')
            print(res.stdout.decode())
            raise Exception(res.stderr.decode())