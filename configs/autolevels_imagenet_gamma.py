from pathlib import Path

cfg = dict(
    name = Path(__file__).stem,

# Setup
    project = 'autolevels',
    out_dir = '/kaggle/working',
    filetype = 'JPEG',
    #filetype = 'tfrec',  # tfds or wds

# Training
    num_folds = 5,
    use_folds = [0],
    train_on_all = False,
    size = (128, 128),
    loss_weights = [0.6, 0.4],
    add_uniform_noise = 0.75,      # add uniform noise to mask uint8 discretization (p)
    noise_level = 0.03,            # random normal noise (augmentation)
    random_blackpoint_shift = 20,
    augmentation = 'tfms_004',     # ignored if use_batch_tfms
    use_batch_tfms = True,
    n_replicas = 8,
    metrics = ['curve_rmse'],
    no_macro_metrics = True,       # otherwise slow valid + 8-TPU-issue
    bs = 64,
    epochs = 1,
    batch_verbose = 1,
    lr_head = 1e-2,
    one_cycle = True,
    div_factor = 5,                            # default: 25, from Chest14: 1
    pct_start = 0.25,                           # default: 0.3, from Chest14: 0.6, pipeline1: 0
    save_best = 'valid_loss',

# Model
    arch_name = 'mobilevitv2_100',
    scale_output_layer = 1.0,
    use_gem = False,
    bn_eps = 1e-5,
    rst_path = '/kaggle/input/autolevels-modelbox',
    #rst_name = '',
    optimizer = "Adam",  # Adam, AdamW, SGD
    dropout_ps = [],
)

cfg["tags"] = cfg["name"].split("_")

cfg["curve"] = 'gamma' if 'gamma' in cfg["tags"] else 'beta'

cfg["lr_bn"] = cfg["lr_head"]
cfg["lr_body"] = cfg["lr_head"]
if cfg["epochs"] == 1: cfg["batch_verbose"] = 10
