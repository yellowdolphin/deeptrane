import os
import sys
from pathlib import Path
import argparse
import importlib

import yaml


class DotDict(dict):
    """dot.notation access to dictionary attributes
    Refer: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def save_yaml(filepath, content, width=120):
    with open(filepath, "w") as f:
        yaml.dump(content, f, width=width)

def load_yaml(filepath):
    with open(filepath, "r") as f:
        content = yaml.safe_load(f)
    return content

class Config(DotDict):
    def __init__(self, config_file):
        if config_file.endswith('.yaml'):
            cfg = load_yaml(config_file)
        else:
            sys.path.append(str(Path(config_file).parent))
            cfg = importlib.import_module(Path(config_file).stem).cfg
        super().__init__(cfg)

    def update(self, cfg):
        if isinstance(cfg, str) or isinstance(cfg, pathlib.Path):
            cfg = self.__class__(cfg)
        super().update(cfg)

    def save_yaml(self, filename=None, width=120):
        if filename is None:
            os.makedirs(self.out_dir, exist_ok=True)
            filename = Path(self.out_dir)/"cfg.yaml"
        save_yaml(filename, dict(self), width=width)


parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config_file", help="config file path")
parser.add_argument("-M", "--mode", default='train', help="mode type")
#parser.add_argument("-S", "--stage", default=0, help="stage")
#parser.add_argument("-W", "--wei_dir", default='.', help="test weight dir")

parser_args, _ = parser.parse_known_args(sys.argv)

print("[ √ ] Config file:", parser_args.config_file)
print("[ √ ] Mode:", parser_args.mode)

cfg = Config('configs/defaults')
if parser_args.config_file: 
    cfg.update(parser_args.config_file)

cfg.mode = parser_args.mode
cfg.save_yaml()

# Consistency checks
if cfg.frac != 1: assert do_class_sampling, 'frac w/o class_sampling not implemented'
if cfg.rst_name is not None:
    rst_file = Path(rst_path)/f'{rst_name}.pth'
    assert rst_file.exists(), f'{rst_file} not found'  # fail early