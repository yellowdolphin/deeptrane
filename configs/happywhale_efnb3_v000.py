from pathlib import Path

cfg = dict(
    name = Path(__file__).stem,

# Setup
    project = 'happywhale',
    out_dir = '/kaggle/working',
    image_root = '/kaggle/input/happywhale-cropped-dataset-yolov5-ds/train_images/train_images',
    filetype = 'jpg',

# Training
    species = ['dusky_dolphin', 'blue_whale', 'melon_headed_whale', 'southern_right_whale'],
#               'fin_whale', 'bottlenose_dolphin', 'spotted_dolphin', 'sei_whale'],
    num_folds = 2,
    use_folds = [0],
    train_on_all = False,
    #add_new_valid = 0,  # ignored if pudae_valid
    pudae_valid = True,
    no_train_singlets = True,
    negative_thres = 0.0,
    size = (256, 256),
    multilabel = False,
    use_albumentations = True,
    augmentation = 'tfms_016',
    num_tpu_cores = 8,
    no_macro_metrics = True,  # otherwise slow valid + 8-TPU-issue
    bs = 64,
    n_acc = 1,
    epochs = 60,
    batch_verbose = None,
    lr_head = 3e-4,
    one_cycle = True,
    div_factor = 3,                            # default: 25, from Chest14: 1
    pct_start = 0.3,                           # default: 0.3, from Chest14: 0.6, pipeline1: 0
    save_best = 'mAP5',
    use_gem = False,
    arcface = 'ArcMarginProduct',
    #feature_size = 64,  # FC-pooling for nfnet_l1, efnv2_s: 16 (128²), 64 (256²), 144 (384²), 256 (512²)
    channel_size = 512,
    arcface_s = 45,
    arcface_m = 0.2,
    bn_eps = 1e-5,
    rst_path = '/kaggle/input/happywhale-deeptrane-classifier-rst',
    rst_name = 'happywhale_efnv2_v004_v39_init',
    optimizer = "AdamW",  # Adam, AdamW, SGD
    wd = 5e-2
)

cfg["tags"] = cfg["name"].split("_")
if 'pretrain' in cfg["tags"]: cfg["num_folds"] = 50
if 'cait' in cfg["tags"]: 
    cfg["arch_name"] = 'cait_xs24_384'
elif 'nfnet' in cfg["tags"]:
    cfg["arch_name"] = 'eca_nfnet_l1'
elif 'efnv2' in cfg["tags"]:
    cfg["arch_name"] = 'tf_efficientnetv2_s_in21ft1k'
elif 'efnb3' in cfg["tags"]:
    cfg["arch_name"] = 'tf_efficientnet_b3_ns'
cfg["lr_bn"] = cfg["lr_head"]
cfg["lr_body"] = cfg["lr_head"]
if cfg["epochs"] == 1: cfg["batch_verbose"] = 10
