from pathlib import Path

cfg = dict(
    name = Path(__file__).stem,

# Setup
    project = 'happywhale',
    out_dir = '/kaggle/working',

# Training
    size = (64, 256),  # WARNING: input images must be pre-scaled, --rect must be set and not --image-weights
    n_bg_images = 0,
    num_folds = 2,
    use_folds = [0],
    train_on_all = False,
    lr = 1.8e-2,

# Aug
    hflip = True,

# Inference
    test_images_path = '/kaggle/input/jpeg-happywhale-384x384/test_images-384-384/test_images-384-384',
)

# Examples for dependent (inferred) settings
cfg["tags"] = cfg["name"].split("_")
