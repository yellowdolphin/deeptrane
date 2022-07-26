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
    "exclude_multiimage_studies": True,
    "use_albumentations": True,
    "n_replicas": 1,
    "bs": 32,
    "n_acc": 2,
    "epochs": 10,
    "batch_verbose": 20,
    "lr_head": 3e-4,
    "save_best": 'mAP',
    "use_gem": True,
    "dropout_ps": [0.6],
    "use_aux_loss": False,
    "seg_weight": 0.4,
    "add_hidden_layer": False,
    "bn_eps": 1e-8,
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
cfg["augmentation"] = 'tfms_' + cfg["name"][13:16]
