from pathlib import Path

cfg = dict(
    name = Path(__file__).stem,

# Setup
    project = 'cassava',
    dataset = 'cassava-leaf-disease-classification',
    out_dir = '/kaggle/working',
    use_custom_training_loop = False,

# Training
    seed = 42,
    num_folds = 4,
    use_folds = [0],
    train_on_all = False,
    size = (256, 256),
    bs = 128,
    n_acc = 1,
    epochs = 10,
    batch_verbose = None,
    multilabel = False,
    optimizer = "Adam",  # Adam, AdamW, SGD
    wd = 5e-2,
    lr = 3e-4,
    one_cycle = True,
    div_factor = 5,                            # default: 25, from Chest14: 1
    pct_start = 0.25,                           # default: 0.3, from Chest14: 0.6, pipeline1: 0
    lr_min = 0.05,
    metrics = ['acc'],
    save_best = 'acc',

# Augmentation
    normalize = None,
    use_albumentations = True,
    augmentation = 'cassava_tf_aug',
    use_batch_tfms = False,

# Model
    arch_name = 'resnet18',
    pool = 'avg',
    dropout_ps = [],
    #arcface = 'ArcMarginProduct',
    #feature_size = 64,  # FC-pooling for nfnet_l1, efnv2_s: 16 (128²), 64 (256²), 144 (384²), 256 (512²)
    #channel_size = 2048,  # embedding size (set automatically if deotte)
    #arcface_s = 30,
    #arcface_m = 0.3,
    #subcenters = 3,
    #adaptive_margin = False,
    #margin_min = 0.18,
    #margin_max = 0.60,

# Restart
    rst_path = '/kaggle/input/cassava-deeptrane-rst',
    #rst_name = '',
)

cfg["tags"] = cfg["name"].split("_")
cfg["pretrained"] = 'noisy_student' if 'v1' in cfg["arch_name"] else 'imagenet21k-ft1k'
#cfg["lr_head"] = cfg["lr"]
#cfg["lr_bn"] = cfg["lr_head"]
#cfg["lr_body"] = cfg["lr_head"]
if cfg["epochs"] == 1: cfg["batch_verbose"] = 1
