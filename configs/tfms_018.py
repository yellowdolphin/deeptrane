import albumentations as alb
from albumentations.pytorch import ToTensorV2 as ToTensor

size = dict(height=384, width=384)  # will be updated by cfg.size
defaults = dict(border_mode=4, value=None, interpolation=1)
distort_limit = 0.5

cfg = dict(
    train_tfms = alb.Compose([
        alb.RandomResizedCrop(**size, scale=[0.8, 1], ratio=(0.75, 1.333), interpolation=1),
        alb.ShiftScaleRotate(p=0.75, shift_limit=0.06, scale_limit=0.3, rotate_limit=22.5, **defaults),
        #alb.HorizontalFlip(p=0.5),
        alb.OneOf(p=0.75, transforms=[
            alb.OpticalDistortion(distort_limit=distort_limit, **defaults),
            alb.GridDistortion(distort_limit=distort_limit, **defaults),
        ]),
        alb.RandomBrightnessContrast(p=1.0, brightness_limit=0.32, contrast_limit=0.32),
        alb.OneOf([
            alb.CLAHE(p=0.15),
            alb.Equalize(p=0.35),
        ], p=0.5),
        alb.CoarseDropout(p=0.75, max_holes=2, max_height=128, max_width=32),
        alb.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225], 
            max_pixel_value=255.0, 
            p=1.0),
        ToTensor(),
    ]),
    test_tfms = alb.Compose([
        alb.Resize(**size, interpolation=1),
        alb.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225], 
            max_pixel_value=255.0, 
            p=1.0),
        ToTensor(),
    ]),
)
