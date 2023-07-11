from pathlib import Path

cfg = dict(
    name = Path(__file__).stem,
    DEBUG = False,
    xla_metrics = False,
    xla_nightly = False,

# Data Partitioning
    train_on_all = False,
    num_folds = 5,
    use_folds = [0],

# Image params
    datasets = [],
    gcs_paths = [],
    image_root = 'data/images',
    size = (64, 64),

# Dataloader
    multilabel = False,
    augmentation = 'tfms_004',
    use_dp = False,  # use DP (deprecated, better use DDP via torchrun train.py)
    n_replicas = 1,
    bs = 8,
    steps_per_execution = 1,  # TF only, increase for performance (check callbacks, training behavior)
    epochs = 1,
    batch_verbose = None,
    do_class_sampling = False,
    frac = 1.0,
    use_batch_tfms = False,
    fake_data = False,
    deviceloader = None,
    num_workers = None,

# Optimizer, Scheduler
    optimizer = 'AdamW',
    n_acc = 1,
    lr_head = 1e-4,
    lr_bn = 1e-4,
    lr_body = 1e-4,
    betas = (0.9, 0.999),
    one_cycle = True,
    div_factor = 1,                            # default: 25, from Chest14: 1
    pct_start = 0.3,                           # default: 0.3, from Chest14: 0.6, pipeline1: 0
    reduce_on_plateau = False,                 # else use Step/MultiStep
    step_lr_after = None,
    step_lr_factor = 1.0,
    save_best = None,                          # metric name or None

# Model
    use_timm = True,           # timm or torchvision pretrained models
    arch_name = 'resnet18',
    use_aux_loss = False,
    seg_weight = 0.4,
    add_hidden_layer = False,   # add hidden_layer from efficientnet to smp model
    pool = 'avg',
    dropout_ps = [0.5],
    lin_ftrs = [],
    scale_output_layer = 1.0,   # modify initialization of the output layer weights
    bn_head = False,
    bn_eps = 1e-5,
    bn_momentum = 0.1,
    wd = 0.05,                  # default 1e-2
    freeze_head = False,

    rst_path = '.',
    rst_name = None,
    reset_opt = False,          # don't load optimizer/scheduler state dicts
    out_dir = 'output',

    xla = False,

    ema = False,
    muliscale = False,
    tta = False,
)

### Examples for dependent (inferred) settings

cfg["tags"] = cfg["name"].split("_")
if 'pretrain' in cfg["tags"]: cfg["num_folds"] = 50
if 'cait' in cfg["tags"]: 
    cfg["arch_name"] = 'cait_xs24_384'
elif 'nfnet' in cfg["tags"]:
    cfg["arch_name"] = 'eca_nfnet_l1'
