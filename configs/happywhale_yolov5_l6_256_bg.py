from pathlib import Path

cfg = dict(
    name = Path(__file__).stem,

# Setup
    project = 'happywhale',
    out_dir = '/kaggle/working',

# Training
    size = 256,
    bs = 32,
    epochs = 40,
    rectangular = False,
    multiscale = True,
    class_weights = True,
    n_bg_images = 220,
    num_folds = 2,
    use_folds = [0],
    train_on_all = False,
    lr = 1.8e-2,

# Aug
    hflip = True,

# Validation
    val_iou_thres = 0.60,
    val_conf_thres = 0.005,
    val_max_det = 1,

# Restart
    rst_path = None,
    rst_name = None,

# Inference
    test_images_path = '/kaggle/input/jpeg-happywhale-384x384/test_images-384-384/test_images-384-384',
)

# Examples for dependent (inferred) settings
cfg["tags"] = cfg["name"].split("_")
cfg["cache_images"] = cfg["size"] < 384  # OOR for size=384, 35904 images
if 'l6' in cfg["tags"]: cfg["pretrained"] = 'hub/yolov5l6.pt'
