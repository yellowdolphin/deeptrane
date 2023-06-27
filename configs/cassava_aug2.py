import albumentations as alb
from albumentations.pytorch import ToTensorV2 as ToTensor

size = dict(height=384, width=384)  # will be updated by cfg.size

cfg = dict(
    train_tfms = alb.Compose([
        alb.RandomResizedCrop(**size, scale=[0.8, 1], ratio=(0.75, 1.333), interpolation=1),
        alb.Transpose(p=0.5),
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.ShiftScaleRotate(
            rotate_limit=90,
            p=0.5),
        alb.HueSaturationValue(
            hue_shift_limit=0.2, 
            sat_shift_limit=0.2, 
            val_shift_limit=0.2, 
            p=0.5
        ),
        alb.RandomBrightnessContrast(
            brightness_limit=(-0.1,0.1), 
            contrast_limit=(-0.1, 0.1), 
            p=0.5
        ),
        alb.CoarseDropout(p=0.5),
        ToTensor(),
    ]),
    test_tfms = alb.Compose([
        alb.Resize(**size, interpolation=1),
        ToTensor(),
    ]),
)
