import albumentations as alb
from albumentations.pytorch import ToTensorV2 as ToTensor

size = dict(height=384, width=384)

cfg = dict(
    train_tfms = alb.Compose([
        alb.RandomResizedCrop(**size, scale=(0.25, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, p=1.0),
        alb.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, interpolation=1, border_mode=0, value=0, p=0.25),
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.OneOf([
            alb.MotionBlur(p=.2),
            alb.MedianBlur(blur_limit=3, p=0.1),
            alb.Blur(blur_limit=3, p=0.1),
        ], p=0.25),
        alb.OneOf([
            alb.CLAHE(clip_limit=2),
            alb.IAASharpen(),
            alb.IAAEmboss(),
            alb.RandomBrightnessContrast(),            
        ], p=0.25),
        alb.CoarseDropout(p=0.25, max_holes=8, max_height=32, max_width=32),
        ToTensor(),
        ]),
    test_tfms = alb.Compose([
        alb.Resize(**size, interpolation=1),
        ToTensor(),
    ]),
)
