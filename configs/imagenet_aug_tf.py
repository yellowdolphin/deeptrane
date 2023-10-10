cfg = dict(
    hflip            = True,
    vflip            = False,
    transpose        = False,
    random_grayscale = None,
    random_crop      = 0.3,  # max pct to cut off 
    rotate           = None,
    hue              = None,
    saturation       = None,
    contrast         = None,
    brightness       = None,
    target_maps      = [lambda target: target - 1],
)
