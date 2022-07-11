from pathlib import Path
from glob import glob
from time import perf_counter

import numpy as np
from numpy.random import default_rng
import pandas as pd
from PIL import Image
from torch import nn


def init(cfg):
    "Update cfg with project- or context-specific logic."
    if 'yolov5' in cfg.tags:
        # Train on bboxes from several datasets.
        # Metadata for the current competition is specified here, external data in `add_bboxes`.
        # BG images from external datasets are added in `filter_bg_images`.
        cfg.competition_path = Path('/kaggle/input/happy-whale-and-dolphin')
        cfg.image_root = Path('/kaggle/input')
        scaled_size = 128 if max(cfg.size) <= 128 else 256 if max(cfg.size) <= 256 else 384
        cfg.subdirs = [(f'jpeg-happywhale-{scaled_size}x{scaled_size}/'
                        f'train_images-{scaled_size}-{scaled_size}/'
                        f'train_images-{scaled_size}-{scaled_size}')]
        if max(cfg.size) > 384:
            cfg.subdirs = ['happy-whale-and-dolphin/train_images']
        cfg.filetype = 'jpg'
        cfg.meta_csv = cfg.competition_path / 'train.csv'
        #cfg.dims_csv = None
    else:
        # support 2-class image, 4-class study classifier, aux_loss, yolov5
        cfg.competition_path = Path('/kaggle/input/happy-whale-and-dolphin')
        if 'colab' in cfg.tags:
            cfg.competition_path = Path('/content/gdrive/MyDrive/siimcovid/siim-covid19-detection')
        if cfg.image_root:
            cfg.filetype = cfg.filetype or 'png'  # use png for all cropped+scaled images
        else:
            scaled_size = 128 if max(cfg.size) <= 128 else 256 if max(cfg.size) <= 256 else 384
            cfg.image_root = Path(f'/kaggle/input/jpeg-happywhale-{scaled_size}x{scaled_size}/'
                                  f'train_images-{scaled_size}-{scaled_size}/'
                                  f'train_images-{scaled_size}-{scaled_size}')
            if max(cfg.size) > 384:
                cfg.image_root = Path('/kaggle/input/happy-whale-and-dolphin/train_images')
            cfg.filetype = 'jpg'
        cfg.meta_csv = cfg.competition_path / 'train.csv'
        cfg.dims_csv = Path('/kaggle/input/happywhale-2022-image-dims/dims.csv')


def extra_columns(cfg):
    if 'yolov5' in cfg.tags:
        return []
    elif 'species' in cfg.tags:
        return ['individual_id']
    else:
        return ['species']


def add_image_id(df, cfg):
    df['image_id'] = df.image.str.split('.').str[0]

    # Also correct misspelled species names
    n_species = df.species.nunique()
    df.species.replace({'kiler_whale': 'killer_whale', 
                        'bottlenose_dolpin': 'bottlenose_dolphin'}, inplace=True)
    if cfg.merge_all_pilot_whales:
        df.species.replace({'globis': 'pilot_whale',
                            'short_finned_pilot_whale': 'pilot_whale',
                            'long_finned_pilot_whale': 'pilot_whale'}, inplace=True)
    elif cfg.merge_short_finned_pilot_whales:
        df.species.replace({'globis': 'short_finned_pilot_whale',
                            'pilot_whale': 'short_finned_pilot_whale'}, inplace=True)
    print(f"{n_species} species merged into {df.species.nunique()} unique")

    return df


def add_category_id(df, cfg):
    if cfg.species:
        cfg.species = set(cfg.species)
        df = df.loc[df.species.isin(cfg.species)].copy()
        print(f"      Selected {len(df)} examples from species", *sorted(cfg.species))
    if 'humpback' in cfg.tags:
        return df.rename(columns={'Id': 'category_id'})
    elif 'yolov5' in cfg.tags:
        return df.rename(columns={'species': 'category_id'})

    if cfg.pudae_valid:
        # Keep individual_id unchanged, sample unpaired, prepare folds.json
        id_counts = df.individual_id.value_counts(sort=False)
        singlets = [i for i, count in id_counts.items() if count == 1]
        doublets = [i for i, count in id_counts.items() if count == 2]
        rest = [i for i, count in id_counts.items() if count > 2]
        rng = default_rng(seed=42)
        rng.shuffle(doublets)
        half = len(doublets) // 2
        train_doublets, valid_doublets = doublets[:half], doublets[half:]
        pct_new = 0.112  # percentage "new_individual" in public test set
        n_valid_singlets = min(len(singlets), int(pct_new * 2 * half / (1 - pct_new)))
        rng.shuffle(singlets)
        train_singlets, valid_singlets = singlets[:-n_valid_singlets], singlets[-n_valid_singlets:]
        if cfg.no_train_singlets:
            train_singlets = []

        image_ids = df.set_index('individual_id').image_id.copy()
        valid_ids = image_ids.loc[valid_singlets + valid_doublets].tolist()
        train_ids = image_ids.loc[train_singlets + train_doublets + rest].tolist()
        shared_doublet_ids = [image_ids.loc[i].iloc[0] for i in valid_doublets]
        
        n_orig = len(df)
        df = df.set_index('image_id').loc[train_ids + valid_ids].reset_index()
        n_new = len(df)
        print(f"      train (unique):    {len(train_ids)}")
        print(f"      train (all):       {len(train_ids) + len(shared_doublet_ids)}")
        print(f"      valid (unique):    {len(valid_ids) - len(shared_doublet_ids)}")
        print(f"      valid (all):       {len(valid_ids)}")
        print(f"      shared doublets:   {len(shared_doublet_ids)}")
        print(f"      dropped singlets:  {n_orig - n_new}")
        print(f"      total unique:      {n_new}")

        cfg.shared_fold = 2  # will be added to valid set in xla_train
        folds = pd.DataFrame({'image_id': df.image_id, 'fold': 1}).set_index('image_id')
        folds.loc[valid_ids] = 0
        folds.loc[shared_doublet_ids] = cfg.shared_fold
        assert folds.index.nunique() == len(folds), f'folds has {len(folds) - folds.index.nunique()} duplicate image_ids'
        cfg.folds_json = Path(cfg.out_dir) / 'folds.json'
        folds.reset_index().to_json(cfg.folds_json)

        return df.rename(columns={'individual_id': 'category_id'})

    # Merge all unpaired individuals as "new_individual"
    cfg.negative_class = 'new_individual'
    if cfg.negative_thres is not None:
        print(f'[ √ ] Predict "{cfg.negative_class}" if top score is below negative_thres {cfg.negative_thres}')
    if not cfg.train_on_all:
        print(f"{df.individual_id.nunique()} individuals")
        id_counts = df.individual_id.value_counts(sort=False)
        new = set([i for i, count in id_counts.items() if count == 1])
        n_new = len(new)
        print(f"{n_new} unpaired -> {cfg.negative_class}")

        # Add random identities to reach 0.112 in valid set
        if cfg.add_new_valid:
            rng = default_rng(seed=42)
            paired_individuals = [i for i, count in id_counts.items() if count > 1]
            add_individuals = rng.choice(paired_individuals, size=cfg.add_new_valid, replace=False)
            new.update(add_individuals)
            print(f'      Declaring all {n_new} unpaired images as "{cfg.negative_class}"')
            print(f'      Adding {cfg.add_new_valid} paired individuals:', *add_individuals[:3].tolist(), '...')
            add_df = df.loc[df.individual_id.isin(set(add_individuals))].copy()
            add_df.drop_duplicates(subset='individual_id', inplace=True)
            df = df.set_index('individual_id').drop(labels=add_individuals).reset_index()
            df = pd.concat([df, add_df], ignore_index=True)

        df.loc[df.individual_id.isin(new), 'individual_id'] = cfg.negative_class
        print(f"{df.individual_id.nunique() - 1} paired individuals after unpaired's name change")

    return df.rename(columns={'individual_id': 'category_id'})


def add_bboxes(df, cfg):
    if 'yolov5' not in cfg.tags:
        return df

    # bboxes are gathered in dataset "happywhale_annotations".
    suffix = f'_v{cfg.annotations_version}' if cfg.annotations_version else ''
    annotations = pd.read_csv(cfg.image_root / 'happywhale-annotations' / f'happywhale_bboxes{suffix}.csv')
    annotations = annotations['image_id x_min y_min x_max y_max height width'.split()].copy()
    assert 'bbox' not in df.columns
    assert 'x_min' not in df.columns
    assert 'image_id' in df.columns
    df = df.merge(annotations, on='image_id', how='left' if cfg.keep_unlabelled else 'inner')
    if cfg.keep_unlabelled:
        # This is useful for cropping images based on predicted bboxes
        xyxy_cols = ['x_min', 'y_min', 'x_max', 'y_max']
        n_unlabelled = df[xyxy_cols].isna().any(axis=1).sum()
        df.fillna(value={'x_min': 0, 'y_min': 0, 'x_max': 1, 'y_max':1}, inplace=True)
        for c in xyxy_cols:
            df[c] = df[c].astype(int)
        print(f"      adding 0 0 1 1 bboxes to {n_unlabelled} unlabelled images")
        assert cfg.train_on_all, 'keep_unlabelled makes no sense without train_on_all!'
        cfg.dims_csv = cfg.dims_csv or Path('/kaggle/input/happywhale-2022-image-dims/dims.csv')
        print(f"      reading image dims from {cfg.dims_csv}")
        assert cfg.dims_csv.exists()

    print(f"[ √ ] {len(df)} bboxes added")

    # Check if image subdirs exist
    subdirs = df.image_path.apply(lambda s: str(Path(s).parent)).unique()
    for subdir in subdirs:
        assert (cfg.image_root / subdir).exists(), f'subdir not found: {cfg.image_root / subdir}'

    return df


def filter_bg_images(bg_images, df, cfg):
    if cfg.n_bg_images == 0: 
        return []

    # Add images with false positive predictions (whales have been eliminated with gimp)
    if True:
        assert (cfg.image_root / 'happywhale-bg-images').exists(), 'dataset happywhale_bg_images not found'
        fns = sorted(glob(f'{cfg.image_root}/happywhale-bg-images/*.jpg'))
        bg_images = [Path(fn).relative_to(cfg.image_root) for fn in fns]
        assert len(bg_images) >= cfg.n_bg_images, f"only found {len(bg_images)} bg images"

    # Add boat/buoy BG images from "Boat types recognition" by Clorichel
    # This did not help!
    if False:
        assert (cfg.image_root / 'boat-types-recognition').exists(), 'dataset boat-types-recognition not found'
        subdirs = [f'boat-types-recognition/{cat}' for cat in ('buoy', 'sailboat', 'ferry boat', 'freight boat')]
        fns = []
        for subdir in subdirs:
            fns.extend(sorted(glob(f'{cfg.image_root}/{subdir}/*.jpg')))

        # Filter by file name
        bg_images = [Path(fn).relative_to(cfg.image_root) for fn in fns if not any([
            'symbol' in fn, 'internat' in fn, 'mari' in fn, 'logo' in fn, 'sign' in fn,
            'port' in fn, 'beach' in fn, 'bridge' in fn, 'hungary' in fn, 'mast' in fn,
            'pirate' in fn, 'watercol' in fn, 'skylin' in fn, 'flag' in fn,
            'montage' in fn, 'city' in fn, 'pressioni' in fn])]

        # Exclude "photoshop" attribute (symbols, clipart), surprisingly fast!
        bg_images = [fn for fn in bg_images if "photoshop" not in Image.open(cfg.image_root / fn).info]

    return bg_images


def after_split(df, cfg):
    if not 'yolov5' in cfg.tags: 
        assert cfg.num_folds == 2
        assert len(cfg.use_folds) == 1

    if "new_individual" in cfg.classes and not cfg.train_on_all:
        # put all new_individuals in valid set
        valid_fold = cfg.use_folds[0]
        n_valid = (df.fold == valid_fold).sum()
        new_id = cfg.vocab.transform(['new_individual'])[0]
        new_images = df.category_id == new_id
        df.loc[new_images, 'fold'] = valid_fold
        n_new = new_images.sum()
        print(f'[ √ ] All {n_new} "new_individuals" in valid fold ({n_new / n_valid:.3f})')

    return df


def pooling(cfg, n_features):
    # feature_size for nfnet_l1: 16 (128²), 64 (256²), 144 (384²), 256 (512²)
    if cfg.arcface: return None
    if cfg.feature_size:
        print(f"building FC pooling with weight {n_features * cfg.feature_size, 512}...")
        return nn.Sequential(nn.Dropout2d(p=0.6, inplace=True),
                             nn.Flatten(),
                             nn.Linear(n_features * cfg.feature_size, 512),
                             nn.BatchNorm1d(512))


def bottleneck(cfg, n_features):
    if len(cfg.dropout_ps) == 2:
        print(f"building bottleneck with weight {n_features, 512}")
        return nn.Sequential(nn.Linear(n_features, 512), nn.Dropout(p=cfg.dropout_ps[1]))
