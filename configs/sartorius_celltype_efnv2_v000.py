from pathlib import Path

cfg = {
    "name": Path(__file__).stem,

# Setup
    "project": 'sartorius',
    "out_dir": '..',

# Training
    "use_folds": [0,1,2,3,4],
    "size": (260, 352),
    "multilabel": False,
    "use_albumentations": True,
    "num_tpu_cores": 8,
    "bs": 32,
    "n_acc": 1,
    "epochs": 10,
    "batch_verbose": None,
    "lr_head": 3e-4,
    "save_best": 'mAP',
    "use_gem": True,
    "dropout_ps": [0.6],
    "use_aux_loss": False,
    "seg_weight": 0.4,
    "add_hidden_layer": False,
    "bn_eps": 1e-5,
    "augmentation": 'tfms_013',
    #"rst_path": '/kaggle/input/siimcovid-classifiers-pretrained',
    #"rst_name": 'chest14_efnv2_ep8',
    #"rst_path": '/kaggle/input/siimcovid-classifiers-rst',
    #"rst_name": 'study_efnv2_v001_init',
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
