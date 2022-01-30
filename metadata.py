import os
from pathlib import Path

import numpy as np
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 100)

DEBUG = False


def get_metadata(cfg, class_column='category_id'):
    "Return a DataFrame with labels, image paths, dims, splits required by the datasets."

    if 'pretrain' in cfg.tags:
        competition_path = Path('/kaggle/input/data')
        meta_csv = 'Data_Entry_2017.csv'
        df = pd.read_csv(competition_path / meta_csv)
        df['image_id'] = df['Image Index'].str.split('.').str[0]
    elif 'study' in cfg.tags or 'image' in cfg.tags:
        competition_path = Path('/kaggle/input/siim-covid19-detection')
        if 'colab' in cfg.tags:
            competition_path = Path('/content/gdrive/MyDrive/siimcovid/siim-covid19-detection')
        meta_csv = 'train_image_level.csv'
        df = pd.read_csv(competition_path / meta_csv)
        df['image_id'] = df.id.str.split('_').str[0]
    elif 'sartorius' in cfg.tags:
        competition_path = Path('/kaggle/input/sartorius-cell-instance-segmentation')
        if 'colab' in cfg.tags:
            competition_path = Path('/content/gdrive/MyDrive/sartorius/sartorius-cell-instance-segmentation')
        meta_csv = 'train.csv'
        df = pd.read_csv(competition_path / meta_csv)
        df.rename(columns={'id': 'image_id'}, inplace=True)

        if 'celltype' in cfg.tags:
            gb = df.groupby('image_id')
            grouped_df = gb.head(1).set_index('image_id')
            df = grouped_df.reset_index()
            del gb, grouped_df
        elif 'bbox' in cfg.tags:
            df['annotation'] = gb.annotation.apply(','.join)
            
        class_column = 'cell_type'
    else:
        competition_path = Path('../../data')
        meta_csv = 'deeptrane_test_meta.csv'
        df = pd.read_csv(competition_path / meta_csv)
        df['image_id'] = df.id.str.split('_').str[0]
    if DEBUG: print(df.head(3))

    if 'image' in cfg.tags or ('add_5th_class' in cfg and cfg.add_5th_class):
        df = add_2class_label(df)

    if 'study' in cfg.tags or 'defaults' in cfg.tags:
        # study labels stored in oh_columns and/or 'category_id'
        df = add_study_label(df, cfg, competition_path)
    elif 'pretrain' in cfg.tags:
        df = add_chest14_labels(df, cfg)

    df = add_filename(df, cfg)

    df = maybe_encode_labels(df, cfg, class_column=class_column)

    required_columns = ['image_id', 'image_path', class_column]
    if cfg.multilabel:
        required_columns.extend(cfg.classes)
        # keep 'category_id' for StratifiedKFold
    if cfg.use_aux_loss:
        required_columns.append('label')
    if 'sartorius' in cfg.tags and 'bbox' in cfg.tags:
        required_columns.append('bbox')
    df = df[required_columns].reset_index(drop=True)
    if DEBUG: print(df.head(3))

    if cfg.use_aux_loss:
        # To properly scale segmentation masks, add original image dims
        meta_csv = os.path.join(cfg.image_root.parent, 'meta.csv')
        df = add_image_dims(df, meta_csv)

    df = split_data(df, cfg, class_column=class_column)

    assert df.columns[0] == 'image_path'  # convention
    assert df.columns[1] == class_column  # convention

    return df


def get_class_ids(s):
    "Return list of class_ids from prediction_string `s`"
    a = np.array(s.split()).reshape(-1, 6)
    return set(a[:, 0].tolist())


def get_single_label_class_ids(s):
    s = get_class_ids(s)
    assert len(s) == 1
    return s.pop()


def add_2class_label(metadata, class_column='category_id'):
    # Image labels are all singlelabels!
    metadata[class_column] = metadata.label.apply(get_single_label_class_ids)
    print(f"{len(metadata)} examples")
    if 'image_id'         in metadata: print(f"{metadata.image_id.nunique()} unique image ids")
    if 'StudyInstanceUID' in metadata: print(f"{metadata.StudyInstanceUID.nunique()} unique study ids")
    if DEBUG: print(metadata.head(3))
    return metadata


def add_study_label(metadata, cfg, competition_path, singlelabel_column='category_id'):
    meta_csv = 'train_study_level.csv'
    train_labels_study = pd.read_csv(competition_path / meta_csv)
    train_labels_study['StudyInstanceUID'] = train_labels_study.id.str.split('_').str[0]

    oh_columns = ['Negative for Pneumonia', 'Typical Appearance',
                  'Indeterminate Appearance', 'Atypical Appearance']
    train_labels_study['study_label'] = train_labels_study[oh_columns].to_numpy().argmax(axis=1)

    # Add study_label to images associated with study.
    metadata = metadata.merge(train_labels_study[['StudyInstanceUID', 'study_label'] + oh_columns],
                              on='StudyInstanceUID').drop(columns='boxes')
    if not cfg.use_aux_loss:
        metadata.drop(columns='label', inplace=True)

    # 5-class multilabel classifier: study-labels + image-level "opacity" as 5th class
    if 'add_5th_class' in cfg and cfg.add_5th_class:
        assert cfg.multilabel, '5th study class makes only sense with multilabel=True'
        metadata['Opacity'] = (metadata[singlelabel_column] == 'opacity').astype(int)
        oh_columns.append('Opacity')

    if not cfg.multilabel:
        metadata.drop(columns=oh_columns, inplace=True)

    metadata[singlelabel_column] = metadata.study_label
    metadata.drop(columns='study_label', inplace=True)

    # Filter images to be included in multi-image studies
    if 'exclude_multiimage_studies' in cfg and cfg.exclude_multiimage_studies:
        # Round 1: Only single-image studies
        images_per_study = metadata.groupby('StudyInstanceUID').image_id.count()
        single_image_studies = images_per_study[images_per_study == 1].index
        metadata = metadata.set_index('StudyInstanceUID').loc[single_image_studies]
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
        mask = metadata.image_id.isin(bad_image_ids)
        if DEBUG: print(f"Filtering out {mask.sum()} unlabelled images")
        metadata = metadata.loc[~mask].copy()
        # In multi-image studies, take only one example (rest are duplicates)
        metadata.drop_duplicates(subset='StudyInstanceUID', keep='last', inplace=True)
        # same result with subset=['StudyInstanceUID'] + oh_columns[:4]

    cfg['classes'] = oh_columns

    if DEBUG:
        print(f"{len(metadata)} examples")
        print(metadata.head(3))

    return metadata


def add_chest14_labels(metadata, cfg, singlelabel_column='category_id'):
    labels = metadata['Finding Labels'].str.split('|').apply(set)
    # Skip some irrelevant classes
    #labels = labels.apply(lambda x: x.difference(['No Finding', 'Cardiomegaly']))
    #labels = labels.apply(lambda x: x.difference(['Cardiomegaly']))
    if cfg.multilabel:
        oh_columns = ['Cardiomegaly', 'Emphysema', 'Effusion', 'No Finding', 'Hernia',
                      'Infiltration', 'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax',
                      'Pleural_Thickening', 'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
        for c in oh_columns:
            metadata[c] = labels.apply(set([c]).issubset).astype(int)
        # Add singlelabel column for StratifiedKFold splitting
        metadata[singlelabel_column] = metadata[oh_columns].to_numpy().argmax(axis=1)
        assert 'classes' not in cfg, 'cannot overwrite class names in cfg'
        cfg.classes = oh_columns
    else:
        # Extract single-label instances from multilabel Chest14 dataset
        singlelabel = labels.apply(len) < 2
        labels = labels.apply(lambda x: 'none' if len(x) == 0 else x.pop())
        metadata[singlelabel_column] = labels
        metadata = metadata[singlelabel]

    if DEBUG: print(metadata.head(5))
    return metadata


def add_filename(metadata, cfg):
    if 'defaults' in cfg.tags:
        datasets = ['../../data/images']
        datatype = 'png'
    elif 'pretrain' in cfg.tags:
        datasets = [f'/kaggle/input/data/images_{i:03d}/images' for i in range(1, 13)]
        datatype = 'png'
    elif 'sartorius' in cfg.tags:
        datasets = ['/kaggle/input/sartorius-cell-instance-segmentation/train']
        datatype = 'png'
    elif cfg.size[0] <= 224:
        datasets = ['/kaggle/input/siim-covid19-resized-to-224px-png/train']
        datatype = 'png'
    elif cfg.size[0] <= 512:
        datasets = ['/kaggle/input/siim-covid19-resized-to-512px-png/train']
        datatype = 'png'
    else:
        datasets = ['/kaggle/input/siim-covid19-resized-to-1024px-jpg/train']
        datatype = 'jpg'
    if 'colab' in cfg.tags:
        datasets = [d.replace('/kaggle/input', '/content') for d in datasets]

    # Copy data (colab) and set image_root path
    path = Path(datasets[0])
    assert path.exists(), f"no folder {path}"
    if cfg.xla and False:  # dataloader.show_batch() hangs
        from kaggle_datasets import KaggleDatasets
        gcs_path = KaggleDatasets().get_gcs_path(path.parent.name)
        path = f'{gcs_path}/train'
    cfg.image_root = path
    print("[ √ ] image_root path:", cfg.image_root)

    # Add image_path above image_root to metadata
    if 'pretrain' in cfg.tags:
        file_names = {}
        for dataset in datasets:
            for fn in os.scandir(dataset):
                image_id = fn.name[:-4]
                file_names[image_id] = os.path.join(dataset, fn)
        metadata['image_path'] = metadata.image_id.apply(lambda i: file_names[i])
    else:
        metadata['image_path'] = metadata.image_id + f'.{datatype}'

    return metadata


def maybe_encode_labels(metadata, cfg, class_column='category_id'):
    if cfg.multilabel:
        assert 'classes' in cfg, 'no multilabel class names found in cfg'
        cfg.n_classes = len(cfg.classes)
    else:
        cfg.n_classes = metadata[class_column].nunique()
        max_label, min_label = metadata[class_column].max(), metadata[class_column].min()

        if metadata[class_column].dtype == 'O' or any(  # `or` prevents TypeError
               max_label + 1 > cfg.n_classes,
               min_label < 0):
            from sklearn.preprocessing import LabelEncoder
            vocab = LabelEncoder()
            vocab.fit(metadata[class_column].values)
            metadata[class_column] = vocab.transform(metadata[class_column].values)
            import pickle
            pickle.dump(vocab, open(f'{cfg.out_dir}/vocab.pkl', 'wb'))
            print(f"[ √ ] Label encoder saved to '{cfg.out_dir}/vocab.pkl'")
            # backtrafo: vocab.inverse_transform(x) or vocab.classes_[x]
            # classes:   vocab.classes_
        else:
            print("[ √ ] No label encoding required.")
    print("[ √ ] n_classes:", cfg.n_classes)
    return metadata


def add_image_dims(metadata, meta_csv, id_col='image_id', height_col='dim0', width_col='dim1'):
    "Add original image dimensions from `meta_csv` (`height_col`, `width_col`) to metadata"
    dims = pd.read_csv(meta_csv)
    dims.rename(columns={id_col: 'image_id', height_col: 'height', width_col: 'width'}, inplace=True)
    dims = dims.loc[:, ['image_id', 'height', 'width']]
    n_images = len(metadata)
    metadata = metadata.merge(dims, on='image_id')
    if len(metadata) < n_images:
        print(f"WARNING: dropped {n_images - len(metadata)} images with unknown original dims")
    if DEBUG: print(metadata.head(3))
    return metadata


def split_data(metadata, cfg, class_column='category_id', seed=42):
    if cfg.train_on_all:
        metadata['fold'] = 0
    else:
        if 'folds' in cfg and cfg.folds:
            # Get splitting from existing folds.json
            assert os.path.exists(cfg.folds), f'no file {cfg.folds}'
            folds = pd.read_json(cfg.folds)['image_id', 'fold']
            metadata = metadata.merge(folds, on='image_id')
        else:
            from sklearn.model_selection import StratifiedKFold

            labels = metadata[class_column].values
            skf = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=seed)

            metadata['fold'] = -1
            for fold, subsets in enumerate(skf.split(metadata.index, labels)):
                metadata.loc[subsets[1], 'fold'] = fold

    metadata.set_index('image_id', inplace=True)
    if DEBUG: print(metadata.head(5))
    return metadata
