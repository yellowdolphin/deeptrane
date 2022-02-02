import os
from pathlib import Path

import pandas as pd

from utils.detection import get_bg_images
from utils.general import sizify

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 100)
DEBUG = False


def get_metadata(cfg, project):
    "Return a DataFrame with labels, image paths, dims, splits required by the datasets."
    detection = any(s in cfg.tags for s in 'yolov5 mmdet'.split())

    cfg.competition_path = Path(cfg.competition_path or '../../data')
    print("[ √ ] Competition path:", cfg.competition_path)

    cfg.image_root = cfg.image_root or '../../data/images'
    if 'colab' in cfg.tags:
        cfg.image_root = cfg.image_root.replace('/kaggle/input', '/content')
    cfg.image_root = Path(cfg.image_root)
    assert cfg.image_root.exists(), f"no folder {cfg.image_root}"
    print("[ √ ] image_root path:", cfg.image_root)

    cfg.subdirs = cfg.subdirs or ['.']
    print("[ √ ] subdirs:", cfg.subdirs)

    cfg.filetype = cfg.filetype or 'png'
    print("[ √ ] type:", cfg.filetype)

    cfg.meta_csv = cfg.meta_csv or '../../data/deeptrane_test_meta.csv'
    print("[ √ ] metadata:", cfg.meta_csv)

    df = pd.read_csv(cfg.meta_csv)

    if hasattr(project, 'add_image_id'):
        df = project.add_image_id(df, cfg)
        print("[ √ ] image_id:", *df.image_id.iloc[:2], "...")

    if hasattr(project, 'add_category_id'):
        df = project.add_category_id(df, cfg)
        print("[ √ ] category_id:", *df.category_id.iloc[:2], "...")

    if hasattr(project, 'add_multilabel_cols'):
        df = project.add_multilabel_cols(df, cfg)
    elif 'defaults' in cfg.tags:
        from projects.siimcovid import add_multilabel_cols
        df = add_multilabel_cols(df, cfg)

    df = add_image_path(df, cfg.image_root, cfg.subdirs, cfg.filetype)
    print("[ √ ] image_path:", *df.image_path.iloc[:2], "...")

    df, cfg.bg_images = maybe_drop_bg_images(df, cfg, project)

    df, cfg.n_classes, cfg.classes = maybe_encode_labels(df, cfg)
    print("[ √ ] n_classes:", cfg.n_classes)
    print("[ √ ] classes:", *cfg.classes[:4], '...' if len(cfg.classes) > 4 else '')

    # Check for unecessary columns, put columns in standard order, reset index
    required_columns = ['image_id', 'image_path', 'category_id']
    if cfg.multilabel and not detection:
        required_columns.extend(cfg.classes)
        # keep 'category_id' for StratifiedKFold
    if detection:
        required_columns.append(cfg.bbox_col)
    if hasattr(project, 'extra_columns'):
        required_columns.extend(project.extra_columns(cfg))
    df = df[required_columns].reset_index(drop=True)

    if DEBUG:
        print("metadata after purge:\n", df.head(3))

    df = maybe_add_image_dims(df, cfg)

    if DEBUG:
        print("metadata after maybe_add_image_dims:\n", df.head(3))

    df = split_data(df, cfg)

    assert df.columns[0] == 'image_path'  # dataset convention
    assert df.columns[1] == 'category_id'  # dataset convention

    return df


def add_image_path(df, image_root, subdirs, filetype='png', xla=False):
    # Copy data (colab) and set image_root path
    if xla and False:  # dataloader.show_batch() hangs
        from kaggle_datasets import KaggleDatasets
        gcs_path = KaggleDatasets().get_gcs_path(image_root.parent.name)
        image_root = f'{gcs_path}/{image_root.name}'

    # Add image_path above image_root to df
    if len(subdirs) == 1 and subdirs[0] == '.':
        df['image_path'] = df.image_id + f'.{filetype}'
    elif len(subdirs) == 1:
        df['image_path'] = f'{subdirs[0]}/' + df.image_id + f'.{filetype}'
    else:
        # Scan subdirs for images, map image_ids to subdirs
        fns = {}
        for subdir in subdirs:
            for fn in os.scandir(image_root / subdir):
                if not fn.name.endswith(filetype): continue
                image_id = Path(fn.name).stem
                fns[image_id] = os.path.join(subdir, fn)
        df['image_path'] = df.image_id.apply(lambda i: fns[i])

    return df


def maybe_drop_bg_images(df, cfg, project):
    if 'yolov5' in cfg.tags:
        # Drop BG images, but keep a list of them.
        # Add a random sample of `n_bg_images` BG images to train set, independent of fold.
        # No BG images in validation. `n_bg_images` is hyperparameter (recommendation: 0...10%)
        bg_images = get_bg_images(df, bbox_col=cfg.bbox_col)
        if hasattr(project, 'filter_bg_images'):
            filtered_bg_images = project.filter_bg_images(bg_images, df, cfg)
        df = df.set_index('image_path').drop(bg_images).reset_index()
        if hasattr(project, 'filter_bg_images'): bg_images = filtered_bg_images
        return df, bg_images

    elif 'mmdet' in cfg.tags:
        raise NotImplementedError('Check how mmdet handles BG images!')

    return df, None


def maybe_encode_labels(df, cfg):
    if cfg.multilabel:
        assert 'classes' in cfg, 'no multilabel class names found in cfg'
        return df, len(cfg.classes), cfg.classes

    classes = df.category_id.unique().tolist()
    max_label, min_label = df.category_id.max(), df.category_id.min()

    if df.category_id.dtype == 'O' or any(max_label + 1 > cfg.n_classes, min_label < 0):
        #                           ^-- prevents TypeError
        from sklearn.preprocessing import LabelEncoder
        import pickle
        vocab = LabelEncoder()
        vocab.fit(df.category_id.values)
        df["category_id"] = vocab.transform(df.category_id.values)
        pickle.dump(vocab, open(f'{cfg.out_dir}/vocab.pkl', 'wb'))
        print(f"[ √ ] Label encoder saved to '{cfg.out_dir}/vocab.pkl'")
        # backtrafo: vocab.inverse_transform(x) or vocab.classes_[x]
        # classes:   vocab.classes_
    else:
        print("[ √ ] No label encoding required.")
    return df, len(classes), classes


def maybe_add_image_dims(df, cfg):
    """If missing, add original image dimensions to df as `height`, `width`.

    Original dims can be read from files `cfg.dims_csv`, `cfg.meta_csv`,
    or given as `cfg.original_size`.
    Column names from cfg attrs `dims_id`, `dims_height`, `dims_width`.
    """
    if 'original_size' in cfg and cfg.original_size:
        original_size = sizify(cfg.original_size)  # all labels refer to same original image size
        df['height'] = original_size[0]
        df['width'] = original_size[1]
        return df

    dims_csv = cfg.dims_csv or cfg.meta_csv
    id_col = cfg.dims_id or 'image_id'
    height_col = cfg.dims_height or 'height'
    width_col = cfg.dims_width or 'width'

    if dims_csv == cfg.meta_csv:
        if height_col in df.columns and width_col in df.columns:
            return df.rename(columns={height_col: 'height', width_col: 'width'})
        else:
            return df  # assume, no orig dims needed

    return add_image_dims(df, dims_csv, id_col, height_col, width_col)


def add_image_dims(df, csv, id_col, height_col, width_col):
    "Add original height, width from `csv` (`height_col`, `width_col`) to `df`"
    dims = pd.read_csv(csv)
    required_columns = id_col, height_col, width_col
    if not all(c in dims.columns for c in required_columns):
        print(f"WARNING: missing columns in {csv},\n    need {required_columns}, got {dims.columns}")
        return df  # hope, no orig dims needed

    dims.rename(columns={id_col: 'image_id', height_col: 'height',
                         width_col: 'width'}, inplace=True)
    dims = dims.loc[:, ['image_id', 'height', 'width']]

    n_images = len(df)
    df = df.merge(dims, on='image_id', how='inner')
    if len(df) < n_images:
        print(f"WARNING: dropped {n_images - len(df)} images with unknown original dims")

    return df


def split_data(df, cfg, seed=42):
    if cfg.train_on_all:
        df['fold'] = 0
    else:
        if 'folds' in cfg and cfg.folds:
            # Get splitting from existing folds.json
            assert os.path.exists(cfg.folds), f'no file {cfg.folds}'
            folds = pd.read_json(cfg.folds)['image_id', 'fold']
            df = df.merge(folds, on='image_id')
        elif cfg.n_classes > 1 and not cfg.multilabel:
            from sklearn.model_selection import StratifiedKFold

            labels = df.category_id.values
            skf = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=seed)

            df['fold'] = -1
            for fold, subsets in enumerate(skf.split(df.index, labels)):
                df.loc[subsets[1], 'fold'] = fold

        else:
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=cfg.num_folds, shuffle=True, random_state=seed)

            df['fold'] = -1
            for fold, subsets in enumerate(kf.split(df.index)):
                df.loc[subsets[1], 'fold'] = fold

    df.set_index('image_id', inplace=True)
    if DEBUG: print(df.head(5))

    return df
