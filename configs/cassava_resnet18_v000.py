from pathlib import Path

cfg = dict(
    name = Path(__file__).stem,

# Setup
    project = 'cassava',
    out_dir = '/kaggle/working',
    filetype = 'jpg',
    #filetype = 'tfrec',  # tfds or wds

# Training
    num_folds = 4,
    use_folds = [0],
    train_on_all = False,
    size = (256, 256),
    multilabel = False,
    use_albumentations = True,
    augmentation = 'cassava_aug2',  # aug2 removes Normalize => much better!
    use_batch_tfms = False,
    n_replicas = 8,
    metrics = ['f1', 'macro_f1', 'class_f1'],
    bs = 128,
    n_acc = 1,
    epochs = 10,
    batch_verbose = None,
    lr_head = 1e-3,
    one_cycle = True,
    div_factor = 5,                            # default: 25, from Chest14: 1
    pct_start = 0.25,                           # default: 0.3, from Chest14: 0.6, pipeline1: 0
    save_best = 'valid_loss',

# Model
    arch_name = 'resnet18',
    use_gem = False,
    #arcface = 'ArcMarginProduct',
    #feature_size = 64,  # FC-pooling for nfnet_l1, efnv2_s: 16 (128²), 64 (256²), 144 (384²), 256 (512²)
    #channel_size = 2048,  # embedding size (set automatically if deotte)
    #arcface_s = 30,
    #arcface_m = 0.3,
    bn_eps = 1e-5,
    rst_path = '/kaggle/input/cassava-resnet18-v25-rst',
    #rst_name = '',
    optimizer = "Adam",  # Adam, AdamW, SGD
    wd = 5e-2,
    dropout_ps = [],
)

cfg["tags"] = cfg["name"].split("_")
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
#cfg["folds_json"] = f'/kaggle/input/cassava-folds-json/folds{cfg["num_folds"]}.json'
