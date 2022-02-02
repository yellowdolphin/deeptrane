from pathlib import Path

cfg = {
    "name": Path(__file__).stem,

# Setup
    "project": 'siimcovid',
    "out_dir": '/kaggle/working',

# Training
    "use_folds": [0],
    "size": (384, 384),
    "multilabel": True,
    "exclude_multiimage_studies": True,   # as in v257...261
    "use_albumentations": True,
    "augmentation": "tfms_faster",
    "num_tpu_cores": 8,
    "bs": 8,
    "epochs": 4,
    "batch_verbose": 10,
    "lr_head": 2e-5,
    "save_best": 'mAP',
    "use_gem": True,
    "dropout_ps": [0.6],
    "rst_path": '/kaggle/input/siimcovid-classifiers-pretrained',
    "rst_name": 'chest14_nfnetl1_ep4',
    "optimizer": "AdamW",  # Adam, AdamW, SGD
    "one_cycle": False,
    "step_lr_after": [1, 2],
    "step_lr_factor": 0.1,
}

cfg["tags"] = cfg["name"].split("_")
if 'pretrain' in cfg["tags"]: cfg["num_folds"] = 50
if 'cait' in cfg["tags"]: 
    cfg["arch_name"] = 'cait_xs24_384'
elif 'nfnet' in cfg["tags"]:
    cfg["arch_name"] = 'eca_nfnet_l1'
cfg["lr_bn"] = cfg["lr_head"]
cfg["lr_body"] = cfg["lr_head"]
if cfg["epochs"] == 1: cfg["batch_verbose"] = 10
