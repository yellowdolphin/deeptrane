import albumentations as alb
from albumentations.pytorch import ToTensorV2 as ToTensor

cfg = dict(
    train_tfms = alb.Compose([
        ToTensor(),
    ]),
    test_tfms = alb.Compose([
        ToTensor(),
    ]),
)
