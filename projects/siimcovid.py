# What is better: class or module?
# Write project module similar to a class implementation, then decide.
# import/instanciation: in setup_yolov5 or train after Config.
# pro class: "missing" attr could be avoided by inheriting from a template class.
from pathlib import Path

import pandas as pd

from . import get_single_label_class_ids


def init(cfg):
    "Update cfg with project- or context-specific logic."
    if 'pretrain' in cfg.tags:
        # plain classifier, no aux loss
        cfg.competition_path = Path('/kaggle/input/data')
        cfg.image_root = Path('/kaggle/input/data')
        cfg.subdirs = [f'images_{i:03d}/images' for i in range(1, 13)]
        cfg.meta_csv = cfg.competition_path / 'Data_Entry_2017.csv'
        cfg.dims_csv = None
    else:
        # support 2-class image, 4-class study classifier, aux_loss, yolov5
        cfg.competition_path = Path('/kaggle/input/siim-covid19-detection')
        if 'colab' in cfg.tags:
            cfg.competition_path = Path('/content/gdrive/MyDrive/siimcovid/siim-covid19-detection')
        filetype = 'jpg' if cfg.size[0] > 512 else 'png'
        scaled_size = 224 if cfg.size[0] <= 224 else 512 if cfg.size[0] <= 512 else 1024
        cfg.image_root = Path(f'/kaggle/input/siim-covid19-resized-to-{scaled_size}px-{filetype}/train')
        cfg.meta_csv = cfg.competition_path / 'train_image_level.csv'
        cfg.dims_csv = cfg.image_root.parent / 'meta.csv'
        cfg.dims_height = 'dim0'
        cfg.dims_width = 'dim1'
    if 'yolov5' in cfg.tags:
        cfg.bbox_col = 'boxes'


def extra_columns(cfg):
    ### keep aux_loss as deeptrane feature or project feature??
    if 'use_aux_loss' in cfg and cfg.use_aux_loss:
        return ['label']
    return []


def add_image_id(df, cfg):
    if 'pretrain' in cfg.tags:
        df['image_id'] = df['Image Index'].str.split('.').str[0]
    elif 'study' in cfg.tags or 'image' in cfg.tags:
        df['image_id'] = df.id.str.split('_').str[0]
    return df


def add_category_id(df, cfg):
    "Asserts identical class_ids in each prediction_string in df.labels"
    if 'study' in cfg.tags and cfg.multilabel:
        df['category_id'] = 'multilabel'  # in 4-class study classifier, category_id is not used
    df['category_id'] = df.label.apply(get_single_label_class_ids)
    print(f"{len(df)} examples")
    if 'image_id' in df: print(f"{df.image_id.nunique()} unique image ids")
    if 'StudyInstanceUID' in df: print(f"{df.StudyInstanceUID.nunique()} unique study ids")
    return df


def add_multilabel_cols(df, cfg):
    """Add study-level labels, which can be used as singlelabel or multilabel.

    In singlelabel case, overwrite `category_id` with 4-class label.
    In multilabel case, "opacity" can be added as 5th class, which is then
    ignored in validation.
    """
    if 'image' in cfg.tags: return df
    if 'pretrain' in cfg.tags: return add_chest14_labels(df, cfg)

    meta_csv = 'train_study_level.csv'
    study_metadata = pd.read_csv(cfg.competition_path / meta_csv)
    study_metadata['StudyInstanceUID'] = study_metadata.id.str.split('_').str[0]

    cfg.classes = ['Negative for Pneumonia', 'Typical Appearance',
                   'Indeterminate Appearance', 'Atypical Appearance']

    study_metadata['study_label'] = study_metadata[cfg.classes].to_numpy().argmax(axis=1)

    # Add study_label to images associated with study.
    df = df.merge(study_metadata[['StudyInstanceUID', 'study_label'] + cfg.classes],
                  on='StudyInstanceUID').drop(columns='boxes')
    if not cfg.use_aux_loss:
        df.drop(columns='label', inplace=True)

    # 5-class multilabel classifier: study-labels + image-level "opacity" as 5th class
    if 'add_5th_class' in cfg and cfg.add_5th_class:
        assert cfg.multilabel, '5th study class makes only sense with multilabel=True'
        df['Opacity'] = (df.category_id == 'opacity').astype(int)
        cfg.classes.append('Opacity')

    if not cfg.multilabel:
        df.drop(columns=cfg.classes, inplace=True)

    df['category_id'] = df.study_label
    df.drop(columns='study_label', inplace=True)

    # Filter images to be included in multi-image studies
    if 'exclude_multiimage_studies' in cfg and cfg.exclude_multiimage_studies:
        # Round 1: Only single-image studies
        images_per_study = df.groupby('StudyInstanceUID').image_id.count()
        single_image_studies = images_per_study[images_per_study == 1].index
        df = df.set_index('StudyInstanceUID').loc[single_image_studies]
    else:
        # Filter out badly labeled image_ids
        # Unlabelled duplicates from https://www.kaggle.com/kwk100/siim-covid-19-duplicate-training-images
        bad_image_ids = set('''
        00c1515729a8 2c130ee08736 684230477525 be65d1a22de5 c843d08b49b8 6534a837497d ea2117b53323 eea3a910fa9e
        1ea01196514a 9b1de1c45491 3b6ad60071d4 04cc2f7f4c4b 05c063f5cef5 156cb1f5c689 4c414b793562 a5a364383f34 b121806162c3
        bee62c601ae9 c6e92e59a0ae 72cf260ddf4c 00c1515729a8 6ff218d10741 7f1924880cf8 32222cc776a2 99f3642f50f5 12e97ed89297
        ab55abb953d1 30b18db28900 550f057ee0b0 b5a415f70aa9 2c130ee08736 684230477525 39d52f244db3 611348a721f7 deea6d6f81a5
        2f973757eaa4 1b92142f4362 4cbc17936e7d 582c442e440b 6e9fad584bff 0102b5cac730 b3ffe59e37c7 7626c521cad7 08acae0bf785
        35e398a5a431 6f54e9cbd180 c3a09e8a600d c4b68b29a072 9939f63af4ff e3e2f20e0264 be65d1a22de5 c843d08b49b8 a57bf6dd6119
        9fcbe25a88e0 717ea5155b46 2a7a456d367e 8d4b3609ed92 9872a8a48f23 e738c549fe8e 149356f04849 6d36ffbc7864 ce51b397b1a6
        608d574388ba 6728e11290af 866e3622cb24 df4f1240317e 267a250932bc 49664f078f0e 869476b0763a a39667fe9a81 b97c6b32105e
        ddb051c1233b ef6e312ca719 8093df07a5d0 779f0040d1b2 76c66ee8e58d 66dabc6f972d df2bb22fa871 6e5946091b8a 75b52bec817f
        3577ee4f26c4 a94171e98807 a5bbd30ed109'''.split())
        mask = df.image_id.isin(bad_image_ids)
        df = df.loc[~mask].copy()
        # In multi-image studies, take only one example (rest are duplicates)
        df.drop_duplicates(subset='StudyInstanceUID', keep='last', inplace=True)
        # same result with subset=['StudyInstanceUID'] + cfg.classes[:4]

    return df


def filter_bg_image_ids(bg_image_ids, df):
    "Remove unlabelled duplicate images from `bg_image_ids`"
    bg_image_ids = set(bg_image_ids)
    n_images_per_study = df.groupby('StudyInstanceUID').image_id.count()
    multi_image_studies = set(n_images_per_study.loc[n_images_per_study > 1].index)
    potential_duplicates = set(df.loc[df.StudyInstanceUID.isin(multi_image_studies), 'image_id'])
    potential_duplicates = potential_duplicates.intersection(bg_image_ids)
    print(f'Excluding {len(potential_duplicates)} / {len(bg_image_ids)}'
          ' potentially unlabelled duplicate BG images')
    bg_image_ids = list(bg_image_ids - potential_duplicates)
    print(f'Safe BG images: {len(bg_image_ids)}')
    return bg_image_ids


def filter_bg_images(bg_images, df, cfg):
    "Remove unlabelled duplicate images from `bg_images`"
    if 'pretrain' in cfg.tags:
        return bg_images
    bg_images = set(bg_images)
    n_images_per_study = df.groupby('StudyInstanceUID').image_id.count()
    multi_image_studies = set(n_images_per_study.loc[n_images_per_study > 1].index)
    potential_duplicates = set(df.loc[df.StudyInstanceUID.isin(multi_image_studies), 'image_path'])
    potential_duplicates = potential_duplicates.intersection(bg_images)
    print(f'Excluding {len(potential_duplicates)} / {len(bg_images)}'
          ' potentially unlabelled duplicate BG images')
    bg_images = list(bg_images - potential_duplicates)
    print(f'Safe BG images: {len(bg_images)}')
    return bg_images


def add_chest14_labels(df, cfg):
    labels = df['Finding Labels'].str.split('|').apply(set)
    # Skip some irrelevant classes
    #labels = labels.apply(lambda x: x.difference(['No Finding', 'Cardiomegaly']))
    #labels = labels.apply(lambda x: x.difference(['Cardiomegaly']))
    if cfg.multilabel:
        oh_columns = ['Cardiomegaly', 'Emphysema', 'Effusion', 'No Finding', 'Hernia',
                      'Infiltration', 'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax',
                      'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
        for c in oh_columns:
            df[c] = labels.apply(set([c]).issubset).astype(int)
        # Add singlelabel column for StratifiedKFold splitting
        df['category_id'] = df[oh_columns].to_numpy().argmax(axis=1)
        assert 'classes' not in cfg, 'cannot overwrite class names in cfg'
        cfg.classes = oh_columns
    else:
        # Extract single-label instances from multilabel Chest14 dataset
        singlelabel = labels.apply(len) < 2
        labels = labels.apply(lambda x: 'none' if len(x) == 0 else x.pop())
        df['category_id'] = labels
        df = df[singlelabel]
    return df
