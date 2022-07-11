from pathlib import Path

cfg = dict(
    name = Path(__file__).stem,

# Setup
    project = 'happywhale',
    out_dir = '/kaggle/working',

# Training
    size = 384,
    bs = 24,
    epochs = 40,
    rectangular = False,
    multiscale = True,
    class_weights = False, # True is better for cls/mAP, False is better for bboxes
    n_bg_images = 50,      # generated from images with false positives
    merge_short_finned_pilot_whales = False,
    merge_all_pilot_whales = False,
    anchor_t = 3.0,        # threshold used by autoanchor
    cls = 0.45,            # weight of class loss (default 0.5)
    box = 0.07,            # weight of box loss (default 0.05)
    ar_lowpass = None,     # only train on images with AR <  ar_lowpass
    ar_highpass = None,    # only train on images with AR >= ar_highpass
    num_folds = 5,
    use_folds = [0],
    train_on_all = False,
    lr = 1.8e-2,
    pretrained = None,

# Aug
    hflip = True,
    #hue = 0.015,
    saturation = 0.5,
    #value = 0.4,
    #perspective = 0.9,

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
