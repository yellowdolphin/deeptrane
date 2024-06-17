cfg = dict(
    hflip            = True,
    vflip            = False,
    transpose        = False,
    random_grayscale = None,
    random_crop      = 0.5,  # max pct to cut off 
    rotate           = None,
    hue              = None,
    saturation       = None,
    contrast         = None,
    brightness       = None,
    noise_level      = 0.0,  # random normal noise after resize
    cutout_max       = 0.2,  # max height/length fraction of blotted area
    cutout_color     = 'black_or_white',  # color value, 0...1, or one of "black|white|black_or_white|random|noise"
)
