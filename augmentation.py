import PIL.Image
from config import Config

### Fastai v2 transforms (without _order)
from collections.abc import Iterable


def listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str): return [o]
    if isinstance(o, Iterable): return list(o)
    return [o]


def process_sz(sz):
    sz = listify(sz)
    return tuple(sz if len(sz) == 2 else [sz[0], sz[0]])


def default_crop_size(w, h):
    return [w, w] if w < h else [h, h]


class GeneralCrop(object):
    def __init__(self, size, crop_size=None, resample=PIL.Image.BILINEAR):
        self.resample, self.size = resample, process_sz(size)
        self.crop_size = None if crop_size is None else process_sz(crop_size)

    def default_crop_size(self, w, h):
        return default_crop_size(w, h)

    def __call__(self, image):
        csize = self.default_crop_size(*image.size) if self.crop_size is None else self.crop_size
        return image.transform(self.size, PIL.Image.EXTENT,
                               self.get_corners(*image.size, *csize), resample=self.resample)

    def get_corners(self, w, h):
        return (0, 0, w, h)


class CenterCrop2(GeneralCrop):
    "Zoom into larger dim by `scale`, try to preserve aspect ratio, ouput: square of `size`."
    def __init__(self, size, scale=1.14, resample=PIL.Image.BILINEAR):
        super().__init__(size, resample=resample)
        self.scale = scale

    def default_crop_size(self, w, h):
        max_dim = max(w, h) / self.scale
        return [min(w, max_dim), min(h, max_dim)]

    def get_corners(self, w, h, wc, hc):
        return ((w - wc) // 2, (h - hc) // 2, (w - wc) // 2 + wc, (h - hc) // 2 + hc)


class Transpose():
    def __repr__(self):
        return 'Transpose()'

    def __call__(self, image):
        if isinstance(image, PIL.Image.Image):
            return image.transpose(PIL.Image.TRANSPOSE)
        else:
            return image.transpose(1, 0, 2)


### Transformation Composers
def get_crop_resize(size, test=False,
                    preserve_ar=False, remove_border=False, random_crop=True,
                    zoom_x=1, zoom_y=1, max_random_zoom=1,
                    interpolation=0,
                    **flags):
    H, W = size
    AR = W / H

    from torchvision.transforms import RandomResizedCrop, Resize, CenterCrop

    if preserve_ar:
        if test or not random_crop:
            print(f"CenterCrop2 with scale={zoom_y if remove_border else 1}")
            return [CenterCrop2(size=(W, H),
                                scale=zoom_y if remove_border else 1,
                                resample=interpolation)]

        return [RandomResizedCrop(size=size,
                                  scale=(1 / max_random_zoom, 1),
                                  ratio=(AR, AR),
                                  interpolation=interpolation)]
    else:
        if test or not random_crop:
            size0 = (int(H * zoom_y), int(W * zoom_x)) if remove_border else size
            print(f"Will apply Resize({size0}) and CenterCrop({size})")
            return [Resize(size0, interpolation=interpolation),
                    CenterCrop(size)]
        else:
            return [RandomResizedCrop(size=size,
                                      scale=(1 / max_random_zoom, 1),
                                      ratio=(0.9 * AR, 1.1 * AR),
                                      interpolation=interpolation)]


def get_tfms(cfg, mode='train'):
    "Return tfms defined in module `cfg.augmentation` or construct torchvision tfms from flags."
    flags = Config('configs/aug_defaults')
    if 'augmentation' in cfg:
        flags.update(cfg.augmentation)

    if mode == 'train' and 'train_tfms' in flags:
        for t in flags.train_tfms:
            if hasattr(t, 'height'): t.height = cfg.size[0]
            if hasattr(t, 'width'): t.width = cfg.size[1]
        return flags.train_tfms

    if 'test_tfms' in flags:
        for t in flags.test_tfms:
            if hasattr(t, 'height'): t.height = cfg.size[0]
            if hasattr(t, 'width'): t.width = cfg.size[1]
        return flags.test_tfms

    return get_torchvision_tfms(cfg, flags, mode)


def get_torchvision_tfms(cfg, flags=None, mode='train'):
    "Construct torchvision transforms from `flags`"

    import math
    import warnings

    import torchvision.transforms as TF

    flags = flags or Config('configs/aug_defaults')
    if 'augmentation' in cfg:
        flags.update(cfg.augmentation)

    size = cfg.size
    interpolation = flags.interpolation

    if (mode != 'train') or cfg.use_batch_tfms:
        return TF.Compose([TF.Resize(size, interpolation=interpolation), TF.ToTensor()])

    from torchvision.transforms import (
        RandomApply, CenterCrop,
        RandomHorizontalFlip, RandomVerticalFlip,
        RandomRotation, Pad, RandomPerspective, #RandomEqualize,
        ColorJitter, Normalize, RandomErasing, Grayscale, GaussianBlur)

    tfms = []

    # transpose: train on 90-deg rotated nybg2021 images
    #if size[1] > size[0]:
    #    tfms.append(Transpose())

    # crop, resize
    if not flags.skip_crop_resize:
        tfms += get_crop_resize(size, test=False, **flags)

    # affine transforms, color transforms
    if flags.horizontal_flip:
        tfms.append(RandomHorizontalFlip())
    if flags.vertical_flip:
        tfms.append(RandomVerticalFlip())
    if flags.max_rotate:
        diag = math.sqrt(size[0]**2 + size[1]**2)
        padding = [1 + int(0.5 * (diag - dim)) for dim in size[::-1]]
        tfms.append(Pad(padding, padding_mode='reflect'))
        tfms.append(RandomRotation(flags.max_rotate, interpolation=interpolation, fill=0))
        if flags.p_perspective:
            warnings.filterwarnings('ignore')
            tfms.append(RandomPerspective(p=flags.p_perspective, interpolation=interpolation))
        tfms.append(CenterCrop(size))
    if any((flags.jitter_brightness, flags.jitter_contrast, flags.jitter_saturation, flags.jitter_hue)):
        tfms.append(ColorJitter(brightness=flags.jitter_brightness,
                                contrast=flags.jitter_contrast,
                                saturation=flags.jitter_saturation,
                                hue=flags.jitter_hue))
    if flags.p_grayscale:
        tfms.append(RandomApply([Grayscale(num_output_channels=3)],
                                p=flags.p_grayscale))

    if flags.hist_equalize:
        tfms.append(RandomEqualize(p=flags.hist_equalize))

    if flags.blur:
        tfms.append(RandomApply([GaussianBlur(kernel_size=5)], p=flags.blur))

    # tensorize, normalize
    tfms.append(TF.ToTensor())
    if flags.normalize:
        tfms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    # tensor transforms
    if flags.p_cutout:
        tfms.append(RandomErasing(p=flags.p_cutout,
                                  scale=(0.05, 0.3), ratio=(0.5, 1.4), value=0.93))
    return TF.Compose(tfms)


def blurred_degenerate_image_tf(img):
    import tensorflow as tf

    dtype = tf.float32 if (img.dtype == tf.uint8) else img.dtype

    kernel = (tf.constant([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=dtype, shape=[3, 3, 1, 1]))
    kernel *= 0.07692307692307693  # normalize
    kernel = tf.tile(kernel, [1, 1, 3, 1])  # duplicate over channels

    strides = [1, 1, 1, 1]

    img = tf.expand_dims(tf.cast(img, dtype), 0)
    img = tf.nn.depthwise_conv2d(img, kernel, strides, padding='SAME')
    img = tf.squeeze(img, 0)

    return img


def blend_tf(img1, img2, ratio: float):
    "Interpolate (ratio 0...1) or extrapolate (ratio > 1) between img1 and img2"
    import tensorflow as tf
    #ratio = float(ratio)

    if (img1.dtype == tf.uint8) or (img2.dtype == tf.uint8):
        img1 = ratio * tf.cast(img1, tf.float32) + (1.0 - ratio) * tf.cast(img2, tf.float32)
        return tf.cast(tf.clip_by_value(img1, 0.0, 255.0), tf.uint8)
    else:
        assert img1.dtype == img2.dtype, f'dtype mismatch: {img1.dtype} (img1), {img2.dtype} (img2)'
        return tf.clip_by_value(ratio * img1 + (1.0 - ratio) * img2, 0.0, 1.0)


def adjust_sharpness_tf(img, sharpness_factor: float):
    """Implements the torchvision function with TF Ops
    Args:
        img (tf.Tensor): Image to be adjusted.
        sharpness_factor (float):  How much to adjust the sharpness. Can be
            any non negative number. 0 gives a blurred image, 1 gives the
            original image while 2 increases the sharpness by a factor of 2.

    Returns:
        tf.Tensor: Sharpness adjusted image.
    """
    return blend_tf(img, blurred_degenerate_image_tf(img), sharpness_factor)


def get_tf_tfms(cfg, mode='train'):
    "Return tfms based on flags from file `cfg.augmentation`."
    import tensorflow as tf

    flags = Config(f'configs/{cfg.augmentation}')
    # inputs is dict or tuple. index/keys must accord to data_format!

    if mode == 'train':
        def random_h_or_v_crop(image, size):
            if tf.random.uniform([]) > 0.5:  # crop either horizontally or vertically
                crop = int((1 - flags.random_crop * tf.random.uniform([])) * size[1])
                h, w = size[0], crop
            else:
                crop = int((1 - flags.random_crop * tf.random.uniform([])) * size[0])
                h, w = crop, size[1]
            image = tf.image.random_crop(image, [h, w, 3])
            return tf.image.resize(image, size)


        rotate = tf.keras.layers.RandomRotation(
            factor=(flags.rotate or 0) * 3.1415 / 180,
            fill_mode='reflect',
            #fill_mode='constant', fill_value=1.0,
            interpolation='bilinear')


        def tfms(inputs, targets, sample_weights=None):
            image = inputs[0] if isinstance(inputs, tuple) else inputs['image']
            image = tf.image.resize(image, cfg.size)
            image = tf.image.transpose(image) if flags.transpose and tf.random.uniform([]) < 0.5 else image
            image = tf.image.random_flip_left_right(image) if flags.hflip else image
            image = tf.image.random_flip_up_down(image) if flags.vflip else image
            image = tf.image.random_hue(image, **flags.hue) if flags.hue else image
            image = tf.image.random_saturation(image, *flags.saturation) if flags.saturation else image
            image = tf.image.random_contrast(image, *flags.contrast) if flags.contrast else image
            image = tf.image.random_brightness(image, **flags.brightness) if flags.brightness else image
            image = rotate(image) if (flags.rotate and tf.random.uniform([]) < 0.5) else image
            if flags.random_crop and (tf.random.uniform([]) < 0.75):
                image = random_h_or_v_crop(image, cfg.size)

            # tf.image_random_jpeg_quality broken on kaggle TPUs
            #if flags.random_jpeg_quality and tf.random.uniform([]) < 0.75:
            #    min_quality, max_quality = int(95 * (1 - flags.random_jpeg_quality)), 95
            #    image = tf.image.random_jpeg_quality(image, min_quality, max_quality)

            if flags.random_grayscale and tf.random.uniform([]) < 0.5 * flags.random_grayscale:
                image = tf.image.adjust_saturation(image, 0)

            # no blur in tf or keras libraries:
            #if flags.mean_filter and tf.random.uniform([]) < 0.5 * flags.mean_filter:
            #    image = tfa.image.mean_filter2d(image, filter_shape=5)

            if flags.noise_level:
                rnd_factor = tf.random.uniform(())
                image += cfg.noise_level * rnd_factor * tf.random.normal((*cfg.size, 3))
                image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

            if isinstance(inputs, tuple):
                inputs = (image, *inputs[1:])
            else:
                inputs['image'] = image
            if sample_weights is None:
                return inputs, targets
            return inputs, targets, sample_weights

        return tfms

    else:
        def tfms(inputs, targets, sample_weights=None):
            image = inputs[0] if isinstance(inputs, tuple) else inputs['image']
            image = tf.image.resize(image, cfg.size)
            if isinstance(inputs, tuple):
                inputs = (image, *inputs[1:])
            else:
                inputs['image'] = image
            if sample_weights is None:
                return inputs, targets
            return inputs, targets, sample_weights

        return tfms