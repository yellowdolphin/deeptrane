import albumentations as alb
from albumentations.pytorch import ToTensorV2 as ToTensor

size = dict(height=384, width=384)  # will be updated by cfg.size
# border_mode 0: constant, 1: replicate, 2: reflect_1001, 3: wrap, 4: reflect_101 (default), 5: transparent
# interpolation 0: nearest, 1: bilinear, 2: bicubic, 3: box/area, 4: lanczos
defaults = dict(border_mode=3, value=None, interpolation=1)
distort_limit = 0.5

cfg = dict(
    train_tfms = alb.Compose([
        alb.RandomResizedCrop(**size, scale=[0.5, 1], ratio=(0.75, 1.333), interpolation=1),
        alb.ShiftScaleRotate(p=0.75, shift_limit=0.06, scale_limit=0.3, rotate_limit=22.5, **defaults),
        #alb.HorizontalFlip(p=0.5),
        #alb.OneOf(p=0.75, transforms=[
        #    alb.OpticalDistortion(distort_limit=distort_limit, **defaults),
        #    alb.GridDistortion(distort_limit=distort_limit, **defaults),
        #]),
        #alb.AdvancedBlur(p=0.25),  # not in current version
        alb.GaussianBlur(p=0.25),
        alb.RandomBrightnessContrast(p=1.0, brightness_limit=0.32, contrast_limit=0.32),
        alb.ToGray(p=0.25),
        alb.OneOf([
            alb.CLAHE(p=0.15),
            alb.Equalize(p=0.35),
            alb.RandomRain(p=0.15),
        ], p=0.65),
        #alb.CoarseDropout(p=0.75, max_holes=2, max_height=128, max_width=32),
        ToTensor(),
    ]),
    test_tfms = alb.Compose([
        alb.Resize(**size, interpolation=1),
        ToTensor(),
    ]),
)
