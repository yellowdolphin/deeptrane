from torchvision.transforms.functional import InterpolationMode

cfg = dict(
    max_rotate         = 22.5,
    jitter_brightness  = 0.2,
    jitter_contrast    = 0.2,
    hist_equalize      = 0.3,
    p_cutout           = 0.75,
    p_perspective      = 0.5,
    interpolation      = InterpolationMode('nearest'),
)
# Interpolation depends on library
#                   nearest  lanczos  bilinear  bicubic  box/area  hamming
# PIL                  0        1         2        3        4         5
# albumentations       0                  1        3
# cv2                  0        4         1        2        3
