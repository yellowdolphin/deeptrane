# Mimick TF augmentations
import albumentations as alb
from albumentations.pytorch import ToTensorV2 as ToTensor

size = dict(height=384, width=384)  # will be updated by cfg.size
defaults = dict(border_mode=4, value=None, interpolation=1)

cfg = dict(
    train_tfms = alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.ColorJitter(p=1.0, brightness=0.1, contrast=(0.8, 1.2), saturation=(0.7, 1.0), hue=0.01),
        alb.RandomResizedCrop(**size, scale=[0.6, 1], ratio=(0.6, 1.67), interpolation=1),
        alb.Rotate(p=0.5, limit=5, **defaults),
        alb.ColorJitter(p=1.0, brightness=0, contrast=0, saturation=(0, 0), hue=0),
        alb.MedianBlur(blur_limit=(5, 5), p=0.25),
        ToTensor(),
    ]),
    test_tfms = alb.Compose([
        alb.Resize(**size, interpolation=1),
        ToTensor(),
    ]),
)
