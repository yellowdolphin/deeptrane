from pathlib import Path

cfg = dict(
    name = Path(__file__).stem,

# Setup
    project = 'happywhale',
    out_dir = '/kaggle/working',

# Training
    size = 128,
    bs = 32,
    epochs = 5,
    rectangular = False,
    multiscale = False,
    class_weights = True,
    n_bg_images = 0,
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
    pretrained = '/kaggle/input/happywhale-deeptrane-yolov5-rst/fold0/weights/best.pt',
    #rst_path = '/kaggle/input/happywhale-deeptrane-yolov5-rst/fold0/weights',  # also needs opt.yaml, epochs > finished epochs
    #rst_name = 'best',

# Inference
    test_images_path = '/kaggle/input/jpeg-happywhale-384x384/test_images-384-384/test_images-384-384',
)

# Examples for dependent (inferred) settings
cfg["tags"] = cfg["name"].split("_")
cfg["cache_images"] = cfg["size"] < 384  # OOR for size=384, 35904 images
#if 'l6' in cfg["tags"]: cfg["pretrained"] = 'hub/yolov5l6.pt'
