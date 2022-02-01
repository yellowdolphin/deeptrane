import os
import sys
from pathlib import Path
from shutil import rmtree
from glob import glob
import argparse
import pkg_resources

import yaml
import numpy as np
from config import Config, parser
from utils.general import quietly_run, listify, sizify, autotype
from utils.detection import split_siimcovid_boxes, write_yolov5_labels, write_dataset_yaml
from metadata import get_metadata

DEBUG = True
cd_yolov5 = False

parser.add_argument("--repo")
parser.add_argument("--branch")
parser.add_argument("--wheels", help="local path to wheels (pycocotools, required packages)")

# Read config file and parser_args
parser_args, _ = parser.parse_known_args(sys.argv)
print("[ √ ] Config file:", parser_args.config_file)
cfg = Config('configs/yolov5_setup')
if parser_args.config_file: cfg.update(parser_args.config_file)
cfg.repo = parser_args.repo or cfg.repo
cfg.branch = parser_args.branch or cfg.branch
cfg.wheels = parser_args.wheels or cfg.wheels
cfg.size = sizify(parser_args.size or cfg.size)
cfg.use_folds = parser_args.use_folds or cfg.use_folds
for key, value in listify(parser_args.set):
    autotype(cfg, key, value)

if DEBUG: print(cfg)

# Installs
if os.path.exists('yolov5'):
    rmtree('yolov5')
if cfg.branch:
    quietly_run(f'git clone -b {cfg.branch} {cfg.repo}')
else:
    quietly_run(f'git clone {cfg.repo}')
quietly_run('pip install -r yolov5/requirements.txt')
print(f"[ √ ] YOLOv5 branch {cfg.branch} from {cfg.repo}")
if cfg.wheels:
    whls = glob(f'{cfg.wheels}/pycocotools-*.whl')
    if whls:
        if len(whls) > 1:
            print('WARNING: found several wheels for pycocotools, using', whls[0])
        quietly_run(f'pip install {whls[0]}')
else:
    quietly_run('pip install pycocotools')
print("[ √ ] pycocotools:", pkg_resources.require("pycocotools")[0].version)

if cd_yolov5: os.chdir(yolov5)
data_path = Path(os.getcwd()) / Path('data' if cd_yolov5 else 'yolov5/data')
assert data_path.exists()

# Metadata, labels
metadata = get_metadata(cfg)
if DEBUG: 
    print("First row in metadata:")
    print(metadata.iloc[0])
if 'image' in cfg.tags:
    metadata = split_siimcovid_boxes(metadata, cfg)
else:
    raise NotImplementedError('Implement bbox splitter first!')
if DEBUG: print(metadata.head(5))

# Dump labels for external validation
metadata.to_json(Path(cfg.out_dir) / 'labels.json')

# Create file structure required by YOLOv5 datasets:
#   - Separate folders for train/valid/test images
#   - Symlinks to actual image file location (superfast)
#   - Labels in separate .txt files, one per image (slow)
#
for fold in set(cfg.use_folds):
    path = data_path / f'fold{fold}'
    print(f"Fold {fold} / {len(cfg.use_folds)}")

    metadata['is_valid'] = metadata.fold == fold
    os.makedirs(path / 'labels' / 'train')
    os.makedirs(path / 'images' / 'train')
    os.makedirs(path / 'labels' / 'valid')
    os.makedirs(path / 'images' / 'valid')
    write_dataset_yaml(cfg, path)

    for image_path in metadata.loc[~ metadata.is_valid, 'image_path'].unique():
        os.symlink(cfg.image_root / image_path, path / 'images' / 'train' / Path(image_path).name)

    for image_path in metadata.loc[metadata.is_valid, 'image_path'].unique():
        os.symlink(cfg.image_root / image_path, path / 'images' / 'valid' / Path(image_path).name)

    if cfg.train_on_all:
        image_path = metadata.image_path.iloc[0]  # YOLOv5 needs at least one valid image
        os.symlink(cfg.image_root / image_path, path / 'images' / 'valid' / Path(image_path).name)

    if cfg.n_bg_images:
        print(f"Adding {cfg.n_bg_images} background images without labels to train.")
        for image_path in np.random.choice(cfg.bg_images, size=cfg.n_bg_images, replace=False):
            os.symlink(cfg.image_root / image_path, path / 'images' / 'train' / Path(image_path).name)

    for split in ('train', 'valid'):
        print(f"Writing label txt files for {split}...")
        write_yolov5_labels(metadata, split, path)


# Write hyperparameter yaml file
with open(data_path / 'hyps' / 'hyp.scratch.yaml', 'r') as fp:
    hyp = yaml.safe_load(fp)

hyp.update(dict(lr0=cfg.lr, lrf=0.1, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                scale=0.6, flipud=0.0, fliplr=0.5))

with open(data_path / 'hyps' / 'hyp.yaml', 'w', encoding='utf8') as fp:
    yaml.dump(hyp, fp, default_flow_style=False, allow_unicode=True)

print("[ √ ] YOLOv5 🚀 is ready for launch!")
