from pathlib import Path

cfg = {
    "name": Path(__file__).stem,

# Setup
    "project": 'sartorius',
    "out_dir": '..',

# Training
    "use_folds": [0,1,2,3,4],
    "size": (384, 384),
    "multilabel": True,
    "exclude_multiimage_studies": False,   # as in v257...261
    "augmentation": "tfms_faster",
    "n_replicas": 8,
    "bs": 8,
    "n_acc": 2,
    "epochs": 12,
    "batch_verbose": 200,
    "lr_head": 5e-5,
    "save_best": 'mAP',
    "use_gem": False,
    "dropout_ps": [0.5],
    "use_aux_loss": True,
    "seg_weight": 0.4,
    "add_hidden_layer": False,
    "bn_eps": 1e-3,
    "rst_path": '/kaggle/input/siimcovid-classifiers-pretrained',
    "rst_name": 'chest14_efnv2_ep8',
    "optimizer": "AdamW",  # Adam, AdamW, SGD
}

cfg["tags"] = cfg["name"].split("_")
if 'pretrain' in cfg["tags"]: cfg["num_folds"] = 50
if 'cait' in cfg["tags"]: 
    cfg["arch_name"] = 'cait_xs24_384'
elif 'nfnet' in cfg["tags"]:
    cfg["arch_name"] = 'eca_nfnet_l1'
elif 'efnv2' in cfg["tags"]:
    cfg["arch_name"] = 'tf_efficientnetv2_s_in21ft1k'
cfg["lr_bn"] = cfg["lr_head"]
cfg["lr_body"] = cfg["lr_head"]
if cfg["epochs"] == 1: cfg["batch_verbose"] = 10
