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
        flags.train_tfms[0].height = cfg.size[0]
        flags.train_tfms[0].width = cfg.size[1]
        return flags.train_tfms
    if 'test_tfms' in flags:
        flags.test_tfms[0].height = cfg.size[0]
        flags.test_tfms[0].width = cfg.size[1]
        return flags.test_tfms

    assert not cfg.use_albumentations, 'define albumentations tfms in cfg.augmentation'
    return get_torchvision_tfms(cfg.size, flags, mode, cfg.use_batch_tfms)


def get_torchvision_tfms(size, flags, mode='train', use_batch_tfms=False):
    "Construct torchvision transforms from `flags`"

    interpolation = flags.interpolation

    import math
    import warnings

    from torchvision.transforms import (
        Compose, ToTensor, RandomApply, Resize, CenterCrop,
        RandomHorizontalFlip, RandomVerticalFlip,
        RandomRotation, Pad, RandomPerspective, RandomEqualize,
        ColorJitter, Normalize, RandomErasing, Grayscale)

    if mode == 'train' and use_batch_tfms:
        return ToTensor()

    if mode == 'test':
        return Compose([Resize(size), ToTensor()])

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

    # tensorize, normalize
    tfms.append(ToTensor())
    if flags.normalize:
        tfms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    # tensor transforms
    if flags.p_cutout:
        tfms.append(RandomErasing(p=flags.p_cutout,
                                  scale=(0.05, 0.3), ratio=(0.5, 1.4), value=0.93))
    return Compose(tfms)
