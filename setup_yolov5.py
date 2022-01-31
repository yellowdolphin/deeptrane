import os
import sys
from pathlib import Path
from shutil import rmtree
from glob import glob
import argparse

import yaml
import numpy as np
from config import Config
from utils.general import quietly_run
from utils.detection import split_siimcovid_boxes, write_yolov5_labels, write_dataset_yaml
from metadata import get_metadata

DEBUG = True

parser = argparse.ArgumentParser(description="Command line arguments supersede config file")
parser.add_argument("-c", "--config_file", help="config file for yolov5 setup")
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

# Installs
if os.path.exists('yolov5'):
    rmtree('yolov5')
if cfg.branch:
    quietly_run(f'git clone -b {cfg.branch} {cfg.repo}')
else:
    quietly_run(f'git clone {cfg.repo}')
quietly_run('pip install -r yolov5/requirements.txt')
if cfg.wheels and os.path.exists(f'{cfg.wheels}/pycocotools-2.0.4.tar'):
    whls = glob(f'{cfg.wheels}/pycocotools-*.whl')
    assert whls, f'no wheel for pycocotools found in {cfg.wheels}'
    if len(whls) > 1:
        print('WARNING: found several wheels for pycocotools, using', whls[0])
    quietly_run(f'pip install {whls[0]}')
else:
    quietly_run('pip install pycocotools')

data_path = Path('data')
assert data_path.exists()

# Metadata, labels
metadata = get_metadata(cfg)
if DEBUG: print(metadata.iloc[0])
if 'study' in cfg.tags:
    metadata = split_siimcovid_boxes(metadata, cfg)
else:
    raise NotImplementedError('Implement bbox splitter first!')
if DEBUG: print(metadata.head(5))


# Create file structure required by YOLOv5 datasets:
#   - Separate folders for train/valid/test images
#   - Labels in separate .txt files, one per image
#   - Symlinks to actual image file location
#
for fold in set(cfg.use_folds):
    path = data_path / f'fold{fold}'
    print(f"Fold {fold} / {len(cfg.use_folds)}")

    metadata['is_valid'] = metadata.fold == fold
    write_dataset_yaml(cfg, path)
    os.makedirs(path / 'labels' / 'train')
    os.makedirs(path / 'images' / 'train')
    os.makedirs(path / 'labels' / 'valid')
    os.makedirs(path / 'images' / 'valid')

    for image_path in metadata.loc[~ metadata.is_valid, 'image_path'].unique():
        os.symlink(image_path, path / 'images' / 'train' / Path(image_path).name)

    for image_path in metadata.loc[metadata.is_valid, 'image_path'].unique():
        os.symlink(image_path, path / 'images' / 'valid' / Path(image_path).name)

    if cfg.train_on_all:
        image_path = metadata.image_path.iloc[0]  # YOLOv5 needs at least one valid image
        os.symlink(image_path, path / 'images' / 'valid' / Path(image_path).name)

    if cfg.n_bg_images:
        print(f"Adding {cfg.n_bg_images} background images without labels to train.")
        for image_id in np.random.choice(cfg.bg_image_ids, size=cfg.n_bg_images, replace=False):
            image_path = metadata.loc[image_id, 'image_path']
            os.symlink(image_path, path / 'images' / 'train' / Path(image_path).name)

    for split in ('train', 'valid'):
        write_yolov5_labels(metadata, split, path)


# Write hyperparameter yaml file
with open('data/hyps/hyp.scratch.yaml', 'r') as fp:
    hyp = yaml.safe_load(fp)

hyp.update(dict(lr0=cfg.lr, lrf=0.1, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                scale=0.6, flipud=0.0, fliplr=0.5))

with open('data/hyps/hyp.yaml', 'w', encoding='utf8') as fp:
    yaml.dump(hyp, fp, default_flow_style=False, allow_unicode=True)
