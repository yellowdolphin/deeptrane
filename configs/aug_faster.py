cfg = dict(
    shift_scale_rotate = 0.75,
    max_random_zoom    = 1.30,
    max_shift          = 0.06,
    max_rotate         = 22.5,
    jitter_brightness  = [0.80, 1.20],
    jitter_contrast    = [0.80, 1.20],
    one_of_three       = 0.333,  # OpticalDist|GridDist|CLAHE
    hist_equalize      = 0.2,
    cutout             = 0.75,
    max_height         = 40,
    max_width          = 40,
    max_holes          = 15,
    interpolation      = 1,
    normalize          = False,
)
