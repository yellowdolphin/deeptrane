from pathlib import Path

cfg = dict(
    name = Path(__file__).stem,

# Setup
    project = 'happywhale',
    out_dir = '/kaggle/working',
    image_root = '/kaggle/input/happywhale-jpeg384-unsubmerged/train_images',
    filetype = 'jpeg',
    #filetype = 'wds',  # tfds or wds
    #dataset = 'happywhale-tfrecords-unsubmerged',  # tfds
    #dataset = 'happywhale-wds-unsubmerged',  # wds

# Training
#    species = ['dusky_dolphin', 'blue_whale', 'melon_headed_whale', 'southern_right_whale'],
#               'fin_whale', 'bottlenose_dolphin', 'spotted_dolphin', 'sei_whale'],
    num_folds = 5,
    use_folds = [0],
    train_on_all = False,
    #add_new_valid = 0,  # ignored if pudae_valid
    pudae_valid = False,
    no_train_singlets = True,
    negative_thres = 0.0,
    size = (384, 384),
    multilabel = False,
    use_albumentations = True,
    augmentation = 'tfms_008',
    use_batch_tfms = False,
    n_replicas = 8,
    metrics = ['acc', 'top5'],
    bs = 32,
    n_acc = 2,
    epochs = 30,
    batch_verbose = None,
    lr_head = 3e-3,
    one_cycle = True,
    div_factor = 5,                            # default: 25, from Chest14: 1
    pct_start = 0.25,                           # default: 0.3, from Chest14: 0.6, pipeline1: 0
    save_best = 'acc',
    use_gem = False,
    arcface = 'ArcMarginProduct',
    #feature_size = 64,  # FC-pooling for nfnet_l1, efnv2_s: 16 (128²), 64 (256²), 144 (384²), 256 (512²)
    #channel_size = 2048,  # embedding size (set automatically if deotte)
    arcface_s = 30,
    arcface_m = 0.3,
    bn_eps = 1e-5,
    rst_path = '/kaggle/input/happywhale-deeptrane-classifier-rst',
    #rst_name = 'happywhale_efnv2_v004_v39_init',
    optimizer = "Adam",  # Adam, AdamW, SGD
    wd = 5e-2,
    dropout_ps = [],
    deotte = True,  # only pool + BN before ArcFace
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
elif 'efnb0' in cfg["tags"]:
    cfg["arch_name"] = 'tf_efficientnet_b0_ns'
cfg["lr_bn"] = cfg["lr_head"]
cfg["lr_body"] = cfg["lr_head"]
if cfg["epochs"] == 1: cfg["batch_verbose"] = 10
cfg["folds_json"] = f'/kaggle/input/happywhalefolds-json/folds{cfg["num_folds"]}.json'
