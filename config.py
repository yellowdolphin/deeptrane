import os
import sys
from pathlib import Path
import argparse
import importlib

import yaml


class DotDict(dict):
    """dot.notation access to dictionary attributes

    Reference:
    https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary/23689767#23689767
    """
    __getattr__ = dict.get  # returns None if missing key, don't use getattr() with default!
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
        if isinstance(cfg, str) or isinstance(cfg, Path):
            cfg = self.__class__(cfg)
        super().update(cfg)

    def save_yaml(self, filename=None, width=120):
        if filename is None:
            os.makedirs(self.out_dir, exist_ok=True)
            filename = Path(self.out_dir) / "cfg.yaml"
        save_yaml(filename, dict(self), width=width)

    def __repr__(self):
        s = [f"{self.__class__.__name__}(\n"]
        for k, v in self.items():
            s.append(f"    {k} = '{v}',\n" if isinstance(v, str) else f"    {k} = {v},\n")
        return "".join(s) + ")"


parser = argparse.ArgumentParser(description="Command line arguments supersede config file")

parser.add_argument("-c", "--config_file", help="config file path")
parser.add_argument("-m", "--mode", default='train', help="mode")
parser.add_argument("-f", "--use_folds", nargs="+", type=int, help="cfg.use_folds")
parser.add_argument("-v", "--batch_verbose", help="mbatch frequency of progress outputs")
parser.add_argument("--size", nargs="+", type=int, help="model input size (int) or (height, width)")
parser.add_argument("--dropout_ps", nargs="+", type=float, help="Dropout probabilities for head Dropout layers")
parser.add_argument("--lin_ftrs", nargs="+", type=int, help="None|output dims of linear (FC) head layers before output layer")
parser.add_argument("--betas", nargs=2, type=float, help="Adam (β1, β2) or SGD (μ, 1 - τ) parameters")
parser.add_argument("--set", nargs=2, action='append', help="(key, value) pair for setting (non-iterable) cfg attributes")
