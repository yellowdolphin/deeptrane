import os
import PIL

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
    return tuple(sz if len(sz)==2 else [sz[0], sz[0]])

def default_crop_size(w, h): return [w, w] if w < h else [h, h]

class GeneralCrop(object):
    def __init__(self, size, crop_size=None, resample=PIL.Image.BILINEAR):
        self.resample, self.size = resample, process_sz(size)
        self.crop_size = None if crop_size is None else process_sz(crop_size)

    def default_crop_size(self, w, h): return default_crop_size(w, h)

    def __call__(self, image):
        #if isinstance(image, np.ndarray): image = PIL.Image.fromarray(image)
        #print(f"GeneralCrop: image: {type(image)} image.size: {image.size} self.size: {self.size}")
        csize = self.default_crop_size(*image.size) if self.crop_size is None else self.crop_size
        #print(f"GeneralCrop: csize: {csize}")
        return image.transform(self.size, PIL.Image.EXTENT, 
                               self.get_corners(*image.size, *csize), resample=self.resample)

    def get_corners(self, w, h): return (0, 0, w, h)

class CenterCrop2(GeneralCrop):
    "Zoom into larger dim by `scale`, try to preserve aspect ratio, ouput: square of `size`."
    def __init__(self, size, scale=1.14, resample=PIL.Image.BILINEAR):
        super().__init__(size, resample=resample)
        self.scale = scale

    def default_crop_size(self, w, h):
        max_dim = max(w, h) / self.scale
        #print(w, h, "->", min(w, max_dim), min(h, max_dim))
        return [min(w, max_dim), min(h, max_dim)]

    def get_corners(self, w, h, wc, hc):
        #print(w, h, wc, hc)
        return ((w - wc) // 2, (h - hc) // 2, (w - wc) // 2 + wc, (h - hc) // 2 + hc)

    
class Transpose():
    def __repr__(self):
        return 'Transpose()'
    def __call__(self, image):
        if isinstance(image, PIL.Image.Image):
            return image.transpose(PIL.Image.TRANSPOSE)
        else:
            return image.transpose(1,0,2)


### Transformation Composers
def get_crop_resize(size, test=False, 
                    preserve_ar=False, remove_border=False, random_crop=True, 
                    zoom_x=1, zoom_y=1, max_random_zoom=1,
                    use_albumentations=False,
                    #interpolation="nearest",
                    interpolation=0,
                    **flags):
    H, W = size
    AR = W / H

    if use_albumentations:
        return [Resize(size)]
    if preserve_ar:
        if test or not random_crop:
            print(f"CenterCrop2 with scale={zoom_y if remove_border else 1}")
            return [CenterCrop2(size     = (W, H), 
                                scale    = zoom_y if remove_border else 1,
                                resample = interpolation)]

        return [RandomResizedCrop(size          = size, 
                                  scale         = (1 / max_random_zoom, 1), 
                                  ratio         = (AR, AR),
                                  interpolation = interpolation)]
    else:
        if test or not random_crop:
            size0 = (int(H * zoom_y), int(W * zoom_x)) if remove_border else size
            print(f"Will apply Resize({size0}) and CenterCrop({size})")
            return [Resize(size0, interpolation=interpolation),
                    CenterCrop(size)]
        else:
            return [RandomResizedCrop(size          = size, 
                                      scale         = (1 / max_random_zoom, 1), 
                                      ratio         = (0.9*AR, 1.1*AR),
                                      interpolation = interpolation)]

def get_tfms(cfg, mode='train'):
    "Return tfms defined in module `cfg.augmentation` or construct them from flags."
    flags = Config('configs/aug_defaults')
    if 'augmentation' in cfg:
        flags.update(cfg.augmentation)

    if mode == 'train' and 'train_tfms' in flags:
        return flags.train_tfms
    elif 'test_tfms' in flags:
        return flags.test_tfms

    # Construct transforms from flags
    
    size = cfg.size
    use_albumentations = cfg.use_albumentations
    interpolation = flags.interpolation
    
    if use_albumentations:
        #print('using albumentations')
        from albumentations import (Compose, OneOf, CenterCrop, Resize, RandomResizedCrop,
            RandomCrop, RandomSizedCrop, SmallestMaxSize, Equalize,
            HorizontalFlip, VerticalFlip, Transpose, RandomRotate90, ShiftScaleRotate, 
            RandomBrightnessContrast, OpticalDistortion, GridDistortion, ElasticTransform, CLAHE, CoarseDropout)
        if os.path.exists('/kaggle'):
            from albumentations.pytorch import ToTensorV2 as ToTensor  # not in v0.1.12
        else:
            import albumentations
            print("albumentations:", albumentations.__version__)
            from albumentations.torch import ToTensor
    else:
        from torchvision.transforms import (
            Compose, ToTensor, RandomApply,
            RandomResizedCrop, Resize, CenterCrop, 
            RandomHorizontalFlip, RandomVerticalFlip, 
            RandomRotation, Pad,
            ColorJitter, Normalize, RandomErasing, Grayscale)

    if mode == 'train' and cfg.use_batch_tfms:
        return ToTensor()

    if mode == 'test':
        return Compose([Resize(*cfg.size), ToTensor()])

    tfms = []
    #defaults = dict(border_mode=cv2.BORDER_CONSTANT, value=0, interpolation=flags.interpolation)
    # Also for X-ray, reflection is slightly better than constant border color:
    defaults = dict(border_mode=4, value=None, interpolation=flags.interpolation)
    
    # transpose: train on 90-deg rotated nybg2021 images
    #if size[1] > size[0]:
    #    tfms.append(Transpose())
    
    # crop, resize
    if use_albumentations and not flags.skip_crop_resize:
        tfms.append(RandomResizedCrop(*size, scale=[0.8, 1], interpolation=interpolation))
        if flags.shift_scale_rotate:
            max_scale = flags.max_random_zoom - 1
            tfms.append(ShiftScaleRotate(shift_limit=flags.max_shift, scale_limit=max_scale, 
                                         rotate_limit=flags.max_rotate, p=flags.shift_scale_rotate, **defaults))
    elif not flags.skip_crop_resize:
        tfms += get_crop_resize(size, test=False, **flags)

    # affine transforms, color transforms
    if use_albumentations:
        if flags.horizontal_flip: tfms.append(HorizontalFlip())
        if flags.vertical_flip:   tfms.append(VerticalFlip())
        #if any((flags.jitter_brightness, flags.jitter_contrast)): 
        #    tfms.append(RandomBrightnessContrast(brightness_limit=[x-1 for x in flags.jitter_brightness],
        #                                         contrast_limit  =[x-1 for x in flags.jitter_contrast]))
        if flags.one_of_three:
            distort_limit = 0.7
            tfms.append(OneOf(p=flags.one_of_three, transforms=[
                OpticalDistortion(distort_limit=distort_limit, **defaults),
                GridDistortion(distort_limit=distort_limit, **defaults),
                #CLAHE(),  # try combining CLAHE and EQUALIZE in a OneOf, instead!
            ]))
        if any((flags.jitter_brightness, flags.jitter_contrast)): 
            tfms.append(RandomBrightnessContrast(brightness_limit=[x-1 for x in flags.jitter_brightness],
                                                 contrast_limit  =[x-1 for x in flags.jitter_contrast]))
        #if hist_equalize: tfms.append(Equalize(p=hist_equdistort_limitalize))
        hist_clahe = 0.3
        tfms.append(OneOf(p=hist_clahe + flags.hist_equalize, 
                          transforms=[CLAHE(p=hist_clahe), Equalize(p=flags.hist_equalize)]))
        
        if flags.p_cutout:
            params = ['max_width', 'max_height', 'max_holes']
            cutout_flags = {k: flags[k] for k in params if k in flags}
            tfms.append(CoarseDropout(p=flags.p_cutout, **cutout_flags))

    else:
        if flags.horizontal_flip and not flags.rotate_90: 
            tfms.append(RandomHorizontalFlip())
        if flags.vertical_flip and not flags.rotate_90: 
            tfms.append(RandomVerticalFlip())
        if flags.rotate_90:
            assert size[0] == size[1], f"PilRandomDihedral currently requires square image"
            tfms.append(PilRandomDihedral(p=1))    # Flip+90Rot
        if flags.max_rotate:
            diag = math.sqrt(size[0]**2 + size[1]**2)
            padding = [1 + int(0.5 * (diag - dim)) for dim in size[::-1]]
            tfms.append(Pad(padding, padding_mode='reflect'))
            tfms.append(RandomRotation(flags.max_rotate, interpolation=interpolation, fill=0))
            tfms.append(CenterCrop(size))
        if any((flags.jitter_brightness, flags.jitter_contrast, flags.jitter_saturation, flags.jitter_hue)):
            tfms.append(ColorJitter(brightness = flags.jitter_brightness,
                                    contrast   = flags.jitter_contrast,
                                    saturation = flags.jitter_saturation,
                                    hue        = flags.jitter_hue))
        if flags.p_grayscale:
            tfms.append(RandomApply([Grayscale(num_output_channels=3)],
                                    p=flags.p_grayscale))
            
    # tensorize, normalize
    if not (use_albumentations and flags.normalize):
        # Albumentations cannot normalize tensors, do this in Dataset tensor_transform.
        tfms.append(ToTensor())
    if flags.normalize and not use_albumentations:
        tfms.append(Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    if use_albumentations: 
        return Compose(tfms)

    # torchvision tensor transforms
    if flags.p_cutout:
        tfms.append(RandomErasing(p     = flags.p_cutout, 
                                  scale = (0.1, 0.3), 
                                  ratio = (0.5, 1.4), 
                                  value = 0.93))
    return Compose(tfms)