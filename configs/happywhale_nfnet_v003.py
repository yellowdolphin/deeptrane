from pathlib import Path

cfg = dict(
    name = Path(__file__).stem,

# Setup
    project = 'happywhale',
    out_dir = '/kaggle/working',

# Training
    species = ['beluga'],
    train_on_all = False,
    num_folds = 2,
    use_folds = [0],
    size = (128, 128),
    multilabel = False,
    use_albumentations = True,
    augmentation = 'tfms_015',
    n_replicas = 8,
    no_macro_metrics = True,  # otherwise slow valid + 8-TPU-issue
    bs = 64,
    n_acc = 1,
    epochs = 60,
    batch_verbose = None,
    lr_head = 1e-4,
    one_cycle = False,
    save_best = 'F1',
    use_gem = False,
    feature_size = 16,  # FC-pooling for nfnet_l1: 16 (128²), 64 (256²), 144 (384²), 256 (512²)
    dropout_ps = [0.5],
    loss = None,
    bn_eps = 1e-5,
    rst_path = '/kaggle/input/happywhale-deeptrane-classifier-rst',
    rst_name = 'happywhale_nfnet_v003_init',
    optimizer = "AdamW",  # Adam, AdamW, SGD
    wd = 5e-2  # try 5e-4
)

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
