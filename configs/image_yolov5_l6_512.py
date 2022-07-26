from pathlib import Path

cfg = dict(
    name = Path(__file__).stem,

# Setup
    project = 'siimcovid',
    out_dir = '/kaggle/working',

# Training
    n_bg_images = 440,
    num_folds = 5,
    train_on_all = False,
    use_folds = [0, 1, 2, 3, 4],
    lr = 1.8e-2,

# Inference
    test_images_path = '/kaggle/input/siim-covid19-resized-to-512px-png/test',
)

# Examples for dependent (inferred) settings
cfg["tags"] = cfg["name"].split("_")
cfg["size"] = (512, 512) if "512" in cfg["tags"] else (1024, 1024)
