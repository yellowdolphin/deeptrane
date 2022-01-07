from pathlib import Path

cfg = {
    "name": Path(__file__).stem,
    "use_folds": [0,1,2,3,4],
    "size": (384, 384),
    "multilabel": True,
    "exclude_multiimage_studies": False,   # as in v257...261
    "use_albumentations": True,
    "augmentation": "tfms_faster",
    "num_tpu_cores": 8,
    "bs": 8,
    "n_acc": 2,
    "epochs": 12,
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
cfg["lr_bn"] = cfg["lr_head"]
cfg["lr_body"] = cfg["lr_head"]
