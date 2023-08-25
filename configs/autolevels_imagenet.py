from pathlib import Path

cfg = dict(
    name = Path(__file__).stem,

# Setup
    project = 'autolevels',
    out_dir = '/kaggle/working',
    filetype = 'JPEG',
    #filetype = 'tfrec',  # tfds or wds

# Training
    num_folds = 4,
    use_folds = [0],
    train_on_all = False,
    size = (128, 128),
    multilabel = False,
    #augmentation = 'cassava_aug2',  # aug2 removes Normalize => much better!
    use_batch_tfms = False,
    n_replicas = 8,
    no_macro_metrics = True,  # otherwise slow valid + 8-TPU-issue
    bs = 128,
    #n_acc = 8,
    epochs = 1,
    batch_verbose = 1,
    lr_head = 1e-3,
    one_cycle = True,
    div_factor = 5,                            # default: 25, from Chest14: 1
    pct_start = 0.25,                           # default: 0.3, from Chest14: 0.6, pipeline1: 0
    save_best = 'valid_loss',

# Model
    arch_name = 'tf_efficientnet_b0_ns',
    use_gem = False,
    bn_eps = 1e-5,
    #rst_path = '/kaggle/input/cassava-deeptrane-rst',
    #rst_name = '',
    optimizer = "Adam",  # Adam, AdamW, SGD
    dropout_ps = [],
)

cfg["tags"] = cfg["name"].split("_")

cfg["curve"] = 'gamma' if 'gamma' in cfg["tags"] else 'beta'

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
