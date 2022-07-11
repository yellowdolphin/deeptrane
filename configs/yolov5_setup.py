from pathlib import Path

cfg = dict(
    name = Path(__file__).stem,

# Setup
    repo = 'https://github.com/yellowdolphin/yolov5.git',
    branch = 'siim_aux',
    wheels = '/kaggle/input/pycocotools',
    out_dir = '../..',

# Training
    annotations_version = None,
    size = 32,
    bs = 8,
    epochs = 1,
    cache_images = False,
    rectangular = False,  # (training) if True: size should be max(height, width), AR constant
    multiscale = False,  # (training)
    class_weights = False,  # disables rectangular
    n_bg_images = 0,
    aux_loss = None,
    train_on_all = False,
    num_folds = 5,
    use_folds = [0, 1, 2, 3, 4],
    lr = 1.8e-2,
    use_adam = False,

# Aug
    hflip = True,
    hue = 0.0,
    saturation = 0.0,
    value = 0.0,
    perspective = 0.0,

# Validation
    val_iou_thres = 0.6,
    val_conf_thres = 0.005,
    val_max_det = 1,

# Restart
    arch_name = 'yolov5s',  # ignored if pretrained
    pretrained = None,
    rst_path = None,
    rst_name = None,

# Inference
    test_images_path = '',
)

# Examples for dependent (inferred) settings
cfg["tags"] = cfg["name"].split("_")
