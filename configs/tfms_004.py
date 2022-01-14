import albumentations as alb
from albumentations.pytorch import ToTensorV2 as ToTensor

size = dict(height=384, width=384)
defaults = dict(border_mode=4, value=None, interpolation=1)

cfg = dict(
    train_tfms = alb.Compose([
        alb.RandomResizedCrop(**size, scale=[0.8, 1], ratio=(0.75, 1.333), interpolation=1),
        alb.ShiftScaleRotate(p=0.75, shift_limit=0.06, scale_limit=0.25, rotate_limit=0, **defaults),
        alb.HorizontalFlip(p=0.5),
        alb.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.05, 0.05), contrast_limit=(-0.05, 0.05)),
        ToTensor(),
    ]),
    test_tfms = alb.Compose([
        alb.Resize(**size, interpolation=1),
        ToTensor(),
    ]),
)
