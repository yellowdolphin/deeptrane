# This module is for pytorch training, only. The data pipeline for train_tf is in tf_data.
"""
# Try leverage tensorflow_datasets for pytorch training

- tensorflow_datasets allows to stream a tfds, which already reads multiple tfrec files, decodes, augments, 
  buffers and shuffles them. It yields numpy arrays, which can be iterated over by a pytorch DataLoader.
- On CPU/single-GPU, a custom torch DataLoader that iterates on the tfds seems to work.
- On TPU, the parallel_dataloader can be initialized with some tricks but `next(iter(pl))` hangs forever.
- What is missing is an Op that converts the TF tensors into pytorch ones on the TPU. Sending the data to 
  the CPU host VM as numpy and then feeding them back via torch device loader might not be efficient.

"""

import os
import re
from functools import partial

import numpy as np
import tensorflow as tf
#!pip install tensorflow-addons
import tensorflow_datasets as tfds
from torch.utils.data import DataLoader
from tf_data import count_data_items

AUTO = tf.data.experimental.AUTOTUNE


def arcface_format(posting_id, image, label_group):
    target = label_group
    return posting_id, {'inp1': image, 'inp2': target}, label_group

def arcface_aux_format(posting_id, image, label_group):
    target = label_group[0]
    return posting_id, {'inp1': image, 'inp2': target}, label_group

def arcface_inference_format(posting_id, image, label_group):
    return image, posting_id

def arcface_eval_format(posting_id, image, label_group):
    return image, label_group  # if aux_loss and possible: use only target, no aux_loss in valid


### AUGMENTATION ### -------------------------------------------------------------------------------------------------------------------

def data_augment(posting_id, image, label_group, cfg):
    # What is difference between stateless and normal tf.image ops?
    # -> normal ones use TF1 RNG and are strongly discouraged, stateless require `seed`.
    # Tried, but makes no difference in speed/result, just more boilerplate code!
    image = tf.image.random_flip_left_right(image) if cfg.hflip else image
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_saturation(image, 0.70, 1.0)
    image = tf.image.random_contrast(image, 0.80, 1.20)
    image = tf.image.random_brightness(image, 0.10)
    if cfg.rotate:
        rotate = tf.keras.layers.RandomRotation(
            factor=cfg.rotate * 3.1415 / 180,
            fill_mode='reflect',
            #fill_mode='constant', fill_value=1.0,
            interpolation='bilinear')
        image = rotate(image) if (tf.random.uniform([]) < 0.5) else image

    if cfg.random_crop and (tf.random.uniform([]) < 0.75):
        if tf.random.uniform([]) > 0.5:  # crop either horizontally or vertically
            crop = int((1 - cfg.random_crop * tf.random.uniform([])) * cfg.size[1])
            h, w = cfg.size[0], crop
        else:
            crop = int((1 - cfg.random_crop * tf.random.uniform([])) * cfg.size[0])
            h, w = crop, cfg.size[1]
        image = tf.image.random_crop(image, [h, w, 3])
        image = tf.image.resize(image, cfg.size)

    # tf.image_random_jpeg_quality broken on kaggle TPUs
    #if cfg.random_jpeg_quality and tf.random.uniform([]) < 0.75:
    #    min_quality, max_quality = int(95 * (1 - cfg.random_jpeg_quality)), 95
    #    image = tf.image.random_jpeg_quality(image, min_quality, max_quality)

    if cfg.random_grayscale and tf.random.uniform([]) < 0.5 * cfg.random_grayscale:
        image = tf.image.adjust_saturation(image, 0)

    # no blur in tf or keras libraries:
    #if cfg.mean_filter and tf.random.uniform([]) < 0.5 * cfg.mean_filter:
    #    image = tfa.image.mean_filter2d(image, filter_shape=5)

    return posting_id, image, label_group


def decode_image(image_data, cfg, box=None):
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

    image = tf.image.resize(image, cfg.size)
    if cfg.normalize in ['torch', 'tf', 'caffe']:
        from keras.applications.imagenet_utils import preprocess_input
        image = preprocess_input(image, mode=cfg.normalize)
    else:
        image = tf.cast(image, tf.float32) / 255.0

    return image


def read_labeled_tfrecord(example, cfg):
    "This function parse our images and also get the target variable"

    image_key = 'image_name'
    target_key = 'target'

    LABELED_TFREC_FORMAT = {
        image_key: tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        target_key: tf.io.FixedLenFeature([], tf.int64),
    }
    if cfg.crop_method:
        LABELED_TFREC_FORMAT[cfg.crop_method] = tf.io.FixedLenFeature([4], tf.int64)
    if cfg.aux_loss:
        LABELED_TFREC_FORMAT['species'] = tf.io.FixedLenFeature([], tf.int64)
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    
    bb = tf.cast(example[cfg.crop_method], tf.int32) if cfg.crop_method else None
    # Does not work: replace bbox in tfrec
    #elif cfg.crop_method == 'df':
        # fooboo
        # Tensor("ParseSingleExample/ParseExample/ParseExampleV2:1", shape=(), dtype=string)
        #print(f"example['image_name']: {dir(example['image_name'])}")
        #print(example['image_name'].device) # None
        #print(example['image_name'].value_index) # 1
        #print(example['image_name'].eval()) # requires session
        #print(example['image_name'])
        #bb = tf.map_fn(lambda image_name: bboxes_train.loc[image_name], tf.reshape(example['image_name'], (1,)), 
        #               parallel_iterations=None, back_prop=False,
        #               swap_memory=False, infer_shape=True, name=None, fn_output_signature=None)
        #print(tf.io.parse_tensor(example['image_name'], tf.string))
        #print(type(example['image_name'])) # tensorflow.python.framework.ops.Tensor
        #print(repr(example['image_name']))
        #image_name = example['image_name']
        #bb = bboxes_train[image_name.ref()]  # KeyError: <Reference wrapping <tf.Tensor 'ParseSingleExample/ParseExample/ParseExampleV2:1' shape=() dtype=string>>
        #bb = get_train_bbox(image_name)  #  TypeError: Tensor is unhashable. Instead, use tensor.ref() as the key.
        # with .ref(): KeyError: <Reference wrapping <tf.Tensor 'image_name:0' shape=() dtype=string>>


    image = decode_image(example['image'], cfg, bb)
    
    #target = tf.one_hot(tf.cast(example['label_group'], tf.int32), depth=N_CLASSES)
    target = tf.cast(example[target_key], tf.int32)
    aux_target = tf.cast(example['species'], tf.int32) if cfg.aux_loss else None            
    label_group = (target, aux_target) if cfg.aux_loss else target

    return example[image_key], image, label_group


def load_dataset(filenames, cfg, ordered=False):
    "This function loads TF Records and returns tensors"
    
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False  # faster
    
    # GPU is allocated by tf.data.TFRecordDataset, tf.config.set_visible_devices, ...
    # By default the entire device mem is allocated at once.
    # Solution A: make GPU invisible
    tf.config.set_visible_devices([], 'GPU')
    assert len(tf.config.list_logical_devices('GPU')) == 0, "failed disabling GPU for tf"

    # Solution B: enable "memory_growth" on device
    #tf.config.experimental.set_memory_growth(device=tf.config.list_physical_devices('GPU')[0], enable=True)
    # -> InternalError: CUDA runtime implicit initialization on GPU:0 failed. Status: out of memory

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    #dataset = dataset.cache()
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(partial(read_labeled_tfrecord, cfg=cfg), num_parallel_calls=AUTO)
    return dataset


def get_training_dataset(filenames, cfg, bs=2):
    print("load_dataset...")
    dataset = load_dataset(filenames, cfg, ordered=False)
    dataset = dataset.map(partial(data_augment, cfg=cfg), num_parallel_calls=AUTO)
    dataset = dataset.map(arcface_aux_format if cfg.aux_loss else arcface_format, num_parallel_calls=AUTO)
    dataset = dataset.map(lambda posting_id, image, label_group: (image, label_group))
    #dataset = dataset.repeat()      ### try to live without
    dataset = dataset.shuffle(2048)
    #dataset = dataset.shuffle(256)   ### DEBUG
    dataset = dataset.batch(bs, drop_remainder=True)
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_val_dataset(filenames, cfg, bs=2):
    dataset = load_dataset(filenames, cfg, ordered=True)
    #dataset = dataset.map(partial(data_augment, cfg=cfg), num_parallel_calls=AUTO)
    dataset = dataset.map(arcface_aux_format if cfg.aux_loss else arcface_format, num_parallel_calls=AUTO)
    dataset = dataset.map(lambda posting_id, image, label_group: (image, label_group))
    dataset = dataset.batch(bs)
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_eval_dataset(filenames, get_targets=True, bs=2):
    dataset = load_dataset(filenames, ordered=True)
    dataset = dataset.map(arcface_eval_format, num_parallel_calls=AUTO)
    if not get_targets:
        dataset = dataset.map(lambda image, target: image)
    dataset = dataset.batch(bs)
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_test_dataset(filenames, get_names=True, bs=2):
    dataset = load_dataset(filenames, ordered=True)
    dataset = dataset.map(arcface_inference_format, num_parallel_calls=AUTO)
    if not get_names:
        dataset = dataset.map(lambda image, posting_id: image)
    dataset = dataset.batch(bs)
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_tf_datasets(cfg, use_fold):
    # Splitting
    # Number of tfrec files must be multiple of cfg.num_folds.
    # All images in one tfrec belong to either train or valid.
    TRAINING_FILENAMES   = [x for i, x in enumerate(cfg.train_files) if i % cfg.num_folds != use_fold]
    VALIDATION_FILENAMES = [x for i, x in enumerate(cfg.train_files) if i % cfg.num_folds == use_fold]
    cfg.NUM_TRAINING_IMAGES   = count_data_items(TRAINING_FILENAMES)
    cfg.NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
    cfg.NUM_TEST_IMAGES = count_data_items(cfg.test_files) if cfg.test_files else 0
    cfg.test_files = cfg.test_files or []

    print(f"              train    valid    test")
    print(f"tfrec files   {len(TRAINING_FILENAMES):<8} {len(VALIDATION_FILENAMES):<8} {len(cfg.test_files)}")
    print(f"images        {cfg.NUM_TRAINING_IMAGES:<8} {cfg.NUM_VALIDATION_IMAGES:<8} {cfg.NUM_TEST_IMAGES}")

    train_ds = get_training_dataset(TRAINING_FILENAMES, cfg, bs=cfg.bs)
    valid_ds = get_val_dataset(VALIDATION_FILENAMES, cfg, bs=cfg.bs)

    print("train_ds, valid_ds assigned, calling tfds.as_numpy()...")
    train_dl, valid_dl = tfds.as_numpy(train_ds), tfds.as_numpy(valid_ds)

    return train_dl, valid_dl


class TFDataLoader(DataLoader):
    def __init__(self, tfdl, n_examples, n_replica=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tfdl = tfdl
        self.n_examples = int(n_examples)
        self.n_replica = int(n_replica)
        
    def __iter__(self):
        return self._tfdl.__iter__()
    
    def __len__(self):
        return self.n_examples // (self.batch_size * self.n_replica)


def get_dataloaders(cfg, use_fold, xm):

        ds_train, ds_valid = get_tf_datasets(cfg, use_fold)

        train_loader = TFDataLoader(
            ds_train,
            n_examples = cfg.NUM_TRAINING_IMAGES,
            n_replica = cfg.n_replicas,
            dataset = None, #train_dl._ds if hasattr(train_dl, '_ds') else train_dl,
            batch_size  = cfg.bs,
            #num_workers = 0 if cfg.n_replicas > 1 else cpu_count(),
            #pin_memory  = True,
            drop_last   = True,
            shuffle     = False)

        valid_loader = TFDataLoader(
            ds_valid,
            n_examples = cfg.NUM_VALIDATION_IMAGES,
            n_replica = cfg.n_replicas,
            dataset = None, #valid_dl._ds if hasattr(valid_dl, '_ds') else valid_dl,
            batch_size  = cfg.bs,
            #num_workers = 0 if cfg.n_replicas > 1 else cpu_count(),
            #pin_memory  = True,
            drop_last   = False,
            shuffle     = False)
