from pathlib import Path

cfg = dict(
    name = Path(__file__).stem,
    DEBUG = False,

# Data Partitioning
    train_on_all = False,
    fake_data = False,
    num_folds = 5,
    use_folds = [0],

# Image params
    size = (64, 64),

# Dataloader
    multilabel = False,
    use_albumentations = True,
    n_replicas = 1,
    bs = 8,
    epochs = 1,
    batch_verbose = None,
    do_class_sampling = False,
    frac = 1.0,
    use_batch_tfms = False,

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
    use_gem = False,            # GeM pooling
    dropout_ps = [0.5],
    lin_ftrs = [],
    bn_eps = 1e-5,
    bn_momentum = 0.1,
    wd = 0.05,                  # default 1e-2

    rst_path = '.',
    rst_name = None,
    reset_opt = False,          # don't load optimizer/scheduler state dicts
    out_dir = '../../temp',

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
