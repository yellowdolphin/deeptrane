import os
import sys
from pathlib import Path
from shutil import rmtree, copytree, copy2
from glob import glob
import pkg_resources
import importlib
from multiprocessing import cpu_count

import yaml
import numpy as np
from config import Config, parser
from utils.general import quietly_run, listify, sizify, autotype
from utils import detection
from utils.detection import write_yolov5_labels, write_dataset_yaml
from metadata import get_metadata

DEBUG = False
cd_yolov5 = False

parser.add_argument("--repo")
parser.add_argument("--branch")
parser.add_argument("--wheels", help="local path to wheels (pycocotools, required packages)")
parser.add_argument("--run", "--launch", action="store_true", help="run YOLOv5 after setup")

# Read config file and parser_args
parser_args, _ = parser.parse_known_args(sys.argv)
print("[ √ ] Config file:", parser_args.config_file)
cfg = Config('configs/yolov5_setup')
if parser_args.config_file: cfg.update(parser_args.config_file)
cfg.repo = parser_args.repo or cfg.repo
cfg.branch = parser_args.branch or cfg.branch
cfg.wheels = parser_args.wheels or cfg.wheels
cfg.use_folds = parser_args.use_folds or cfg.use_folds
cfg.epochs = parser_args.epochs or cfg.epochs
cfg.size = sizify(parser_args.size or cfg.size)
cfg.keep_unlabelled = cfg.keep_unlabelled or False
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

if cd_yolov5: os.chdir('yolov5')
data_path = Path(os.getcwd()) / Path('data' if cd_yolov5 else 'yolov5/data')
assert data_path.exists()

# Import project (code, constant settings)
project = importlib.import_module(f'projects.{cfg.project}') if cfg.project else None
if project:
    print("[ √ ] Project:", cfg.project)
    project.init(cfg)

# Metadata, labels
metadata = get_metadata(cfg, project)
if DEBUG:
    print(f"Row 1 / {len(metadata)} in metadata after get_metadata():")
    print(metadata.iloc[0])
if hasattr(project, 'get_yolov5_labels'):
    metadata = project.get_yolov5_labels(metadata, cfg)
else:
    metadata = detection.get_yolov5_labels(metadata, cfg)
if DEBUG: 
    print(f"Row 1 / {len(metadata)} in metadata after get_yolov5_labels():")
    print(metadata.iloc[0])

# Dump labels for external validation
cfg.out_dir = Path(cfg.out_dir)
os.makedirs(cfg.out_dir, exist_ok=True)
metadata.to_json(cfg.out_dir / 'labels.json')

# Create file structure required by YOLOv5 datasets:
#   - Separate folders for train/valid images
#   - Symlinks to actual image file location (superfast)
#   - Labels in separate .txt files, one per image (slow)
#
cfg.use_folds = sorted(set(cfg.use_folds))  # duplicate folds would overwrite results
for i, fold in enumerate(cfg.use_folds):
    path = data_path / f'fold{fold}'
    print(f"Preparing data for fold {fold} ({i+1}/{len(cfg.use_folds)} to go)")

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
        print(f"Adding {cfg.n_bg_images} unlabelled background images to train.")
        for image_path in np.random.choice(cfg.bg_images, size=cfg.n_bg_images, replace=False):
            os.symlink(cfg.image_root / image_path, path / 'images' / 'train' / Path(image_path).name)

    for split in ('train', 'valid'):
        print(f"Writing label txt files for {split}...")
        write_yolov5_labels(metadata, split, path)

# Write hyperparameter yaml file
hyp_yaml = data_path / 'hyps' / 'hyp.yaml'
with open(data_path / 'hyps' / 'hyp.scratch.yaml', 'r') as fp:
    hyp = yaml.safe_load(fp)

hyp.update(dict(lr0=cfg.lr, lrf=0.1, 
                hsv_h=cfg.hue or 0.0, hsv_s=cfg.saturation or 0.0, hsv_v=cfg.value or 0.0,
                scale=0.6, flipud=0.0, fliplr=0.5 if cfg.hflip else 0, 
                perspective=cfg.perspective or 0.0,
                anchor_t=cfg.anchor_t or 4.0,
                cls=cfg.cls or 0.5,
                box=cfg.box or 0.05))

with open(hyp_yaml, 'w', encoding='utf8') as fp:
    yaml.dump(hyp, fp, default_flow_style=False, allow_unicode=True)


print("[ √ ] YOLOv5 🚀 is ready for launch!")

if not parser_args.run:
    quit()

assert len(cfg.use_folds) == 1
fold = f'fold{cfg.use_folds[0]}'
os.chdir(data_path.parent)
quietly_run('wandb disabled')

# Prepare restart files
if cfg.rst_path and cfg.rst_name:
    checkpoint_file = cfg.out_dir / fold / 'weights' / f'{cfg.rst_name}.pt'
    if Path(cfg.rst_path).name == 'weights':
        rst_fold = Path(cfg.rst_path).parent.name
        if rst_fold != fold:
            print(f"WARNING: restart from potentially different fold: {rst_fold}")
        copytree(Path(cfg.rst_path).parent, cfg.out_dir / fold, dirs_exist_ok=True)
        assert os.path.exists(checkpoint_file), f'{cfg.rst_name}.pt not found in {cfg.rst_pth}'
        assert os.path.isfile(checkpoint_file), f'{checkpoint_file} is no file'
        assert os.path.exists(cfg.out_dir / fold / 'opt.yaml'), f'opt.yaml missing for resume'
    else:
        os.makedirs(cfg.out_dir / fold / 'weights', exist_ok=True)
        copy2(Path(cfg.rst_path) / f'{cfg.rst_name}.pt', checkpoint_file)
        if (Path(cfg.rst_path) / f'{cfg.rst_name}_opt.yaml').exists():
            copy2(Path(cfg.rst_path) / f'{cfg.rst_name}_opt.yaml', cfg.out_dir / fold / 'opt.yaml')
        elif (Path(cfg.rst_path).parent / 'opt.yaml').exists():
            copy2(Path(cfg.rst_path).parent / 'opt.yaml', cfg.out_dir / fold / 'opt.yaml')
        elif (Path(cfg.rst_path) / 'opt.yaml').exists():
            copy2(Path(cfg.rst_path) / 'opt.yaml', cfg.out_dir / fold / 'opt.yaml')

# Compose YOLOv5 command-line options
weights_or_resume = (f'--resume {checkpoint_file}' if cfg.rst_path and cfg.rst_name else
                     f'--weights {cfg.pretrained}' if cfg.pretrained else
                     f'--weights "" --cfg {cfg.arch_name}.yaml')
size = f'--img {max(cfg.size)}'
rect = '--rect' if cfg.rectangular else ''
batch = f'--batch {cfg.bs}'
epochs = f'--epochs {cfg.epochs}'
hyp = f'--hyp {hyp_yaml.relative_to(data_path.parent)}'
adam = f'--adam' if cfg.use_adam else ''
data = f'--data {fold}/dataset.yaml'
image_weights = f'--image-weights' if cfg.class_weights and (cfg.n_classes > 1) else ''
multi_scale = '--multi-scale' if cfg.multiscale else ''
noautoanchor = '--noautoanchor' if cfg.noautoanchor else ''
project = f'--project {cfg.out_dir}'
name = f'--name {fold}'
workers = f'--workers {cpu_count()}'
exist_ok = '--exist-ok'
val = ('--noval' if cfg.train_on_all else
       f'--val_iou_thres {cfg.val_iou_thres} --val_conf_thres {cfg.val_conf_thres} --val_max_det {cfg.val_max_det}')
cache = '--cache' if cfg.cache_images else ''
aux_loss = f'--aux_loss {cfg.aux_loss} --noautoanchor' if cfg.aux_loss else ''

print(f"\nRunning YOLOv5 on {fold}...")
cml_options = ' '.join([weights_or_resume, size, rect, batch, epochs, hyp, adam, data, image_weights,
                        multi_scale, noautoanchor, project, name, workers, exist_ok, val, cache, aux_loss])
quietly_run(f'python train.py {cml_options}', debug=True)
