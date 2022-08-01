import re
import os
from functools import partial

import numpy as np
import tensorflow as tf

from augmentation import get_tf_tfms

AUTO = tf.data.experimental.AUTOTUNE

# tfrec_format describes the TFRecord features that will be processed and their types.
default_tfrec_format = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'target': tf.io.FixedLenFeature([], tf.int64)}

# data_format maps the features in tfrec_format to default names used by pipeline,
# model, and loss functions.
default_data_format = {
    'image': 'image',
    'target': 'target'}

default_inputs = ['image']
default_targets = ['target']


def count_data_items(filenames):
    "Infer number of items from tfrecord file names."
    pattern = re.compile(r"-([0-9]*)\.")
    n = [int(pattern.search(fn).group(1)) for fn in filenames if pattern.search(fn)]
    if len(n) < len(filenames):
        print(f"WARNING: only {len(n)} / {len(filenames)} urls follow the convention:")
        for fn in filenames:
            print(fn)

    return np.sum(n)


def get_gcs_path(cfg, project=None):
    "Get GCS dataset path from cfg.dataset and kaggle, fallback to lists from project"

    assert cfg.dataset, 'specify either gcs_path or dataset in config'
    gcs_paths = project.gcs_paths if hasattr(project, 'gcs_paths') else {}

    if cfg.cloud == 'kaggle':
        from kaggle_datasets import KaggleDatasets
        from kaggle_web_client import BackendError

        if cfg.private_dataset:
            # Enable GCS for private datasets
            assert os.path.exists(f'/kaggle/input/{cfg.dataset}'), f'Add {cfg.dataset} first!'
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            user_credential = user_secrets.get_gcloud_credential()
            try:
                user_secrets.set_tensorflow_credential(user_credential)
            except NotImplementedError:
                print("tensorflow_gcs_config import failed, probably because of installed tf version")
                print("Using /kaggle/input instead")

        try:
            gcs_path = KaggleDatasets().get_gcs_path(cfg.dataset)
        except ConnectionError:
            print("ConnectionError, using project.gcs_paths")
            gcs_path = gcs_paths[cfg.dataset]
        except BackendError:
            print(f"dataset {cfg.dataset} not in /kaggle/input, using project.gcs_paths")
            gcs_path = gcs_paths[cfg.dataset]
        print("kaggle gcs_path:", gcs_path)

    else:
    
        if cfg.private_dataset:
            from google.colab import auth
            auth.authenticate_user()

        gcs_path = gcs_paths[cfg.dataset]

    return gcs_path


def get_shards(cfg):
    assert cfg.gcs_path, 'need gcs_path for splitting!'
    pattern = cfg.gcs_filter or '*.tfrec*'
    all_files = np.sort(tf.io.gfile.glob(f'{cfg.gcs_path}/{pattern}'))
    n_files = len(all_files)
    assert n_files > 0, f'no tfrec files found at {cfg.gcs_path}'
    return all_files, n_files


def split_by_name(cfg):
    "Split shards into train/valid, based on filename."

    all_files, n_files = get_shards(cfg)

    train_files = [f for f in all_files if Path(f).name.startswith('train')]
    valid_files = [f for f in all_files if Path(f).name.startswith('val')]
    if (n_files > len(train_files) > 0) and (len(train_files) + len(valid_files) == n_files):
        cfg.train_files = train_files
        cfg.valid_files = valid_files


def cv_split(cfg, use_fold):
    "Split shards into train/valid according to cfg.num_folds and `use_fold`."

    assert cfg.num_folds, 'need num_folds in config for shard splitting'
    all_files, n_files = get_shards(cfg)

    train_files, valid_files = [], []
    for i, url in enumerate(all_files):
        if i % cfg.num_folds == use_fold:
            valid_files.append(url)
        else:
            train_files.append(url)
    cfg.train_files = train_files
    cfg.valid_files = valid_files


def show_tfrec_features(url, bboxes=[], text_features=[]):
        try:
            raw_dataset = tf.data.TFRecordDataset(url)
        except InvalidArgumentError:
            print('might need to call cloud_tpu_client.Client.configure_tpu_version()')
            raise

        for raw_record in raw_dataset.take(1):
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            for k, v in example.features.feature.items():
                print(f'{k:30} {str(v)[:10]} {v.ByteSize() / 1024:12.3f} kB')

            if 'image' in example.features.feature:
                bytes_list = example.features.feature['image'].bytes_list
                image = tf.image.decode_jpeg(bytes_list.value[0], channels=3)
                print("image:", image.shape)

            for cm in bboxes:
                if not cm in example.features.feature: continue
                int64_list = example.features.feature[cm].int64_list
                bb = int64_list.value
                print(f'bbox ({cm}):', (bb[3]-bb[1], bb[2]-bb[0]))

            for feature in text_features:
                if feature in example.features.feature:
                    bytes_list = example.features.feature[feature].bytes_list
                    value = bytes_list.value[0]
                    print(f'{feature}:', value.decode('utf-8'))


def decode_image(cfg, image_data, box=None):
    "Decode our images (updated to include crops)"
    
    if box is not None: # and box[0] >= 0 and box[1] >= 0 and box[2] > box[0] and box[3] > box[1]:
        left, top, right, bottom = box[0], box[1], box[2], box[3]
        bbs = tf.convert_to_tensor([top, left, bottom - top, right - left])
        image = tf.io.decode_and_crop_jpeg(image_data, bbs, channels=3)
        # Error handling does not work, try does not catch errors
    else:
        image = tf.image.decode_jpeg(image_data, channels=3)
    
    if cfg.BGR:
        image = tf.reverse(image, axis=[-1])

    if cfg.normalize in ['torch', 'tf', 'caffe']:
        from keras.applications.imagenet_utils import preprocess_input
        image = preprocess_input(image, mode=cfg.normalize)
    else:
        image = tf.cast(image, tf.float32) / 255.0

    return image


def parse_tfrecord(cfg, example):
    """This TFRecord parser extracts the features defined in cfg.tfrec_format.

    TFRecord feature names are mapped to default names by cfg.data_format.
    Default names are understood by the data pipeline, model, and loss function. 
    Returns: (inputs, targets, [sample_weights])

    Default names (keys): 'image', 'mask', 'bbox', 'target', 'aux_target', 'image_id'

    Only keys in cfg.tfrec_format will be parsed.
    """

    example = tf.io.parse_single_example(example, cfg.tfrec_format)
    features = {}

    bbox = tf.cast(example[cfg.data_format['bbox']], tf.int32) if 'bbox' in cfg.data_format else None

    if 'image' in cfg.data_format:
        features['image'] = decode_image(cfg, example[cfg.data_format['image']], bbox)

    if 'mask' in cfg.data_format:
        features['mask'] = decode_image(cfg, example[cfg.data_format['mask']], bbox)

    if 'target' in cfg.data_format:
        features['target'] = tf.cast(example[cfg.data_format['target']], tf.int32)
        #features['target'] = tf.one_hot(features['target'], depth=cfg.n_classes)
    
    if 'aux_target' in cfg.data_format:
        features['aux_target'] = tf.cast(example[cfg.data_format['aux_target']], tf.int32)

    if 'image_id' in cfg.data_format:
        features['image_id'] = example[cfg.data_format['image_id']]

    # tf.keras.model.fit() wants dataset to yield a tuple (inputs, targets, [sample_weights])
    # inputs can be a dict
    inputs = tuple(features[key] for key in cfg.inputs)
    targets = tuple(features[key] for key in cfg.targets)

    return inputs, targets


def get_dataset(cfg, mode='train'):
    filenames = cfg.train_files if mode == 'train' else cfg.valid_files
    if cfg.DEBUG: tf.print(f'urls for mode "{mode}":', filenames)
    shuffle = cfg.shuffle_buffer or 2048 if mode == 'train' else None
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # faster
    tfms = get_tf_tfms(cfg, mode)
    data_format = cfg.data_format  # assume is mode agnostic for now
    bs = cfg.bs
    bs *= cfg.n_replicas
    if cfg.double_bs_in_valid and (mode == 'train'): bs *= 2

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.cache() if cfg.cache_data else dataset
    dataset = dataset.with_options(ignore_order) if shuffle else dataset
    dataset = dataset.map(partial(parse_tfrecord, cfg), num_parallel_calls=AUTO)
    dataset = dataset.map(tfms, num_parallel_calls=AUTO) if tfms else dataset
    dataset = dataset.repeat() if mode == 'train' else dataset
    dataset = dataset.shuffle(shuffle) if shuffle else dataset
    dataset = dataset.batch(bs)
    dataset = dataset.prefetch(AUTO)

    return dataset


def check_data_format(cfg):
    data_format = cfg.data_format
    tfrec_format = cfg.tfrec_format
    inputs = cfg.inputs
    targets = cfg.targets
    tfrec_features = set(tfrec_format.keys())
    required_features = set(data_format.values())
    missing_features = required_features - tfrec_features
    unused_features = tfrec_features - required_features
    if unused_features:
        print("Unused TFRecord features:", unused_features)
    if missing_features:
        print("There are features missing in tfrec_format or data_format is wrong.")
        print("Missing features:", missing_features)
        print("Features in first TFRecord file:")
        show_tfrec_features(get_shards(cfg)[0])
        raise ValueError(f'required features missing in tfrec_format: {missing_features}')

    for inp in cfg.inputs:
        assert inp in cfg.data_format, f'"{inp}" (cfg.inputs) is missing in cfg.data_format'
    for target in cfg.targets:
        assert target in cfg.data_format, f'"{inp}" (cfg.targets) is missing in cfg.data_format'


def configure_data_pipeline(cfg, project):
    cfg.tfrec_format = cfg.tfrec_format or default_tfrec_format
    cfg.data_format = cfg.data_format or default_data_format
    cfg.inputs = cfg.inputs or default_inputs
    cfg.targets = cfg.targets or default_targets

    if cfg.arcface and not hasattr(project, 'inputs'):
        cfg.inputs.append('target')
    if cfg.aux_loss and not hasattr(project, 'targets'):
        cfg.tfrec_format['aux_target'] = tf.io.FixedLenFeature([], tf.int64)
        cfg.targets.append('aux_target')

    if cfg.mode in ['eval', 'inference']:
        for key in ['target', 'aux_target']:
            cfg.tfrec_format.pop(key, None)
        for key in ['target', 'aux_target']:
            cfg.data_format.pop(key, None)
            cfg.inputs.pop(key, None)
            cfg.targets.pop(key, None)

    check_data_format(cfg)
