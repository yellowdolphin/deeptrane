from pathlib import Path

cfg = dict(
    name = Path(__file__).stem,

# Setup
    project = 'autolevels',
    datasets = ['imagenet-1k-tfrecords-ilsvrc2012-part-0', 
                'imagenet-1k-tfrecords-ilsvrc2012-part-1'],
    gcs_paths = [],
    gcs_filters = ['*/*-of-*', '*-of-*'],
    tfrec_filename_pattern = r"-of-([0-9]*)$",
    out_dir = '/kaggle/working',
    use_custom_training_loop = False,

# Training
    seed = 42,
    num_folds = 5,
    use_folds = [0],
    train_on_all = False,
    size = (384, 384),
    multilabel = False,
    metrics = ['curve_rmse'],
    no_macro_metrics = True,  # otherwise slow valid + 8-TPU-issue
    bs = 64,
    steps_per_execution = 1,  # increase for performance (check callbacks, training behavior)
    epochs = 4,
    batch_verbose = 1,
    lr = 1e-3,
    one_cycle = True,
    div_factor = 5,                            # default: 25, from Chest14: 1
    pct_start = 0.25,                           # default: 0.3, from Chest14: 0.6, pipeline1: 0
    lr_min = 1e-6,
    loss_weights = [0.6, 0.4],
    save_best = 'loss',
    add_uniform_noise = 0.75,  # add uniform noise to mask uint8 discretization [bool|float]
    noise_level = 0.03,        # random normal noise (augmentation)
    random_blackpoint_shift = 20,
    augmentation = 'autolevels_aug_tf',

# Model
    arch_name = 'efnv2s',
    bn_eps = 1e-5,
    #rst_path = '/kaggle/input/cassava-deeptrane-rst',
    #rst_name = '',
    optimizer = "Adam",  # Adam, AdamW, SGD
    wd = 5e-2,
    dropout_ps = [],
)

cfg["tags"] = cfg["name"].split("_")

cfg["curve"] = 'gamma' if 'gamma' in cfg["tags"] else 'beta'

cfg["pretrained"] = 'noisy-student' if 'v1' in cfg["arch_name"] else 'imagenet21k-ft1k'
if cfg["epochs"] == 1: cfg["batch_verbose"] = 1
