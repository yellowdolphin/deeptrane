from torchvision.transforms.functional import InterpolationMode

cfg = dict(
    shift_scale_rotate = 0.75,
    skip_crop_resize   = False,
    horizontal_flip    = True,
    vertical_flip      = False,
    max_random_zoom    = 1.30,
    max_shift          = 0.06,
    max_rotate         = 0,
    jitter_brightness  = 0,
    jitter_contrast    = 0,
    jitter_hue         = 0,
    jitter_saturation  = 0,
    hist_equalize      = 0,
    p_grayscale        = 0,
    p_cutout           = 0,
    p_perspective      = 0,
    interpolation      = InterpolationMode('nearest'),
    normalize          = False,
)
# Interpolation depends on library
#                   nearest  lanczos  bilinear  bicubic  box/area  hamming
# PIL/torchvision      0        1         2        3        4         5
# albumentations       0                  1        3
# cv2                  0        4         1        2        3