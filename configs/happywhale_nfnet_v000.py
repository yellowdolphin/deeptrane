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
    augmentation = 'tfms_014',
    n_replicas = 8,
    no_macro_metrics = True,  # otherwise slow valid + 8-TPU-issue
    bs = 64,
    n_acc = 1,
    epochs = 20,
    batch_verbose = None,
    lr_head = 1e-4,
    one_cycle = True,   # False for lr opt
    save_best = 'F1',
    use_gem = False,
    loss = None,
    dropout_ps = [0.6],
    bn_eps = 1e-5,
    rst_path = '/kaggle/input/happywhale-deeptrane-classifier-rst',
    rst_name = 'happywhale_nfnet_v000_init',
    optimizer = "AdamW",  # Adam, AdamW, SGD
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
