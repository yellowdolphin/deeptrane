import albumentations as alb
from albumentations.pytorch import ToTensorV2 as ToTensor

cfg = dict(
    train_tfms = alb.Compose([
        alb.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0, 
            always_apply=True
        ),
        alb.Transpose(p=0.5),
        alb.HorizontalFlip(p=0.5),
        alb.VerticalFlip(p=0.5),
        alb.ShiftScaleRotate(p=0.5),
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
        alb.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            always_apply=True
        ),
        ToTensor(),
    ]),
)
