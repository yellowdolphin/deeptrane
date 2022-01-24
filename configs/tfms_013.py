import albumentations as alb
from albumentations.pytorch import ToTensorV2 as ToTensor

size = dict(height=384, width=384)
defaults = dict(border_mode=4, value=None, interpolation=1)
distort_limit = 1.0

cfg = dict(
    train_tfms = alb.Compose([
        alb.RandomResizedCrop(**size, scale=[0.8, 1], ratio=(0.75, 1.333), interpolation=1),
        alb.ShiftScaleRotate(p=0.75, shift_limit=0.06, scale_limit=0.3, rotate_limit=22.5, **defaults),
        alb.HorizontalFlip(p=0.5),
        alb.OneOf(p=0.75, transforms=[
            alb.OpticalDistortion(distort_limit=distort_limit, **defaults),
            alb.GridDistortion(distort_limit=distort_limit, **defaults),
        ]),
        alb.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
        alb.OneOf([
            alb.CLAHE(p=0.15),
            alb.Equalize(p=0.35),
        ], p=0.5),
        alb.CoarseDropout(p=0.75, max_holes=10, max_height=38, max_width=38),
        ToTensor(),
    ]),
    test_tfms = alb.Compose([
        alb.Resize(**size, interpolation=1),
        ToTensor(),
    ]),
)
