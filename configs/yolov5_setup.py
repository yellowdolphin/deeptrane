from pathlib import Path

cfg = dict(
    name = Path(__file__).stem,

# Setup
    repo = 'https://github.com/yellowdolphin/yolov5.git',
    branch = 'siim_aux',
    wheels = '/kaggle/input/pycocotools',
    out_dir = '../..',
    bbox_col = 'bbox',

# Training
    size = (32, 32),
    n_bg_images = 0,
    aux_loss = None,
    num_folds = 5,
    train_on_all = False,
    use_folds = [0, 1, 2, 3, 4],
    lr = 1.8e-2,

# Inference
    test_images_path = '',
)

# Examples for dependent (inferred) settings
cfg["tags"] = cfg["name"].split("_")
