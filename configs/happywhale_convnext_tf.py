from pathlib import Path

cfg = dict(
    name = Path(__file__).stem,

# Setup
    project = 'happywhale',
    dataset = 'happywhale-tfrecords-unsubmerged',
    out_dir = '/kaggle/working',
    #image_root = '/kaggle/input/happywhale-cropped-dataset-yolov5-ds/train_images/train_images',

# Training
    seed = 142,
    num_folds = 10,
    use_folds = [3],
    train_on_all = False,
    size = (600, 600),
    bs = 8,
    n_acc = 1,
    epochs = 35,
    batch_verbose = 2,
    multilabel = False,
    optimizer = "Adam",  # Adam, AdamW, SGD
    wd = 5e-2,
    lr = 2e-4,
    one_cycle = True,
    div_factor = 5,                            # default: 25, from Chest14: 1
    pct_start = 0.3,                           # default: 0.3, from Chest14: 0.6, pipeline1: 0
    lr_min = 0.05,
    metrics = ['acc', 'top5'],
    save_best = 'acc',

# Augmentation
    normalize = None,
    augmentation = 'happywhale_tf_aug',
    use_batch_tfms = False,

# Model
    arch_name = 'convnext_base_384_in22ft1k',
    pool = 'avg',
    dropout_ps = [0.25],
    arcface = 'ArcMarginProduct',
    subcenters = 2,
    #feature_size = 64,  # FC-pooling for nfnet_l1, efnv2_s: 16 (128²), 64 (256²), 144 (384²), 256 (512²)
    #channel_size = 2048,  # embedding size (set automatically if deotte)
    arcface_s = 30,
    arcface_m = 0.3,
    bn_head = 'batch_norm',
    bn_eps = 1e-5,
    aux_loss = 0.4,

# Restart
    rst_path = '/kaggle/input/happywhale-deeptrane-classifier-rst',
    #rst_name = 'happywhale_efnv2_v004_v39_init',
)

cfg["tags"] = cfg["name"].split("_")
cfg["pretrained"] = 'noisy-student' if 'v1' in cfg["arch_name"] else 'imagenet21k-ft1k'
if cfg["epochs"] == 1: cfg["batch_verbose"] = 1
