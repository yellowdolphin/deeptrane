try:
    from torchvision.transforms.functional import InterpolationMode
except ImportError:
    from configs.torchvision import InterpolationMode

cfg = dict(
    skip_crop_resize   = True,
    max_rotate         = 5,
    jitter_brightness  = 0.1,
    jitter_contrast    = 0.2,
    jitter_saturation  = 0.3,
    jitter_hue         = 0.01,
    hist_equalize      = 0,
    p_cutout           = 0,
    p_perspective      = 0,
    blur               = 0.25,
    interpolation      = InterpolationMode('bilinear'),
)
# Interpolation depends on library
#                   nearest  lanczos  bilinear  bicubic  box/area  hamming
# PIL                  0        1         2        3        4         5
# albumentations       0                  1        3
# cv2                  0        4         1        2        3
