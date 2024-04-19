import os
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd

try:
    import cv2
except ImportError as e:
    if 'cannot open shared object' in str(e):
        print(e, "\ntrying to install ffmpeg libsm6 libext6...")
        # cv2 broken on current kaggle TPU environment
        from utils.general import quietly_run
        quietly_run('apt-get update', debug=False)
        quietly_run('apt-get install ffmpeg libsm6 libxext6 -y', debug=False)
        import cv2
        print("cv2:", cv2.__version__)
    else:
        raise

print("import torch...")
import torch
from torch.utils.data import Dataset
import torchvision
print("[ √ ] torchvision:", torchvision.__version__)
if int(torchvision.__version__.split('.')[1]) >= 15:
    torchvision.disable_beta_transforms_warning()
    import torchvision.transforms.v2 as TT  # new backward compatible API
    from torchvision.transforms.v2.functional import InterpolationMode
else:
    import torchvision.transforms as TT
    from torchvision.transforms.functional import InterpolationMode
from torchvision.io import encode_jpeg, decode_jpeg
from scipy.interpolate import interp1d

try:
    import torchmetrics as tm
    from torchmetrics import MeanSquaredError
except ModuleNotFoundError:
    from utils.general import quietly_run
    # tm requirements insane:
    # avoid replacing torch 2.2.1+cu121 by 2.2.1
    quietly_run('pip install --no-deps torchmetrics==1.3.1 lightning_utilities', debug=False)
    import torchmetrics as tm
    from torchmetrics import MeanSquaredError
print("[ √ ] torchmetrics:", tm.__version__)

try:
    import albumentations as alb
except ModuleNotFoundError:
    from utils.general import quietly_run
    quietly_run('pip install albumentations')
    import albumentations as alb
print("[ √ ] albumentations:", alb.__version__)
from albumentations.augmentations import blur
from albumentations.augmentations.transforms import ImageCompression

try:
    import tensorflow_probability as tfp
except ImportError as e:
    print(e)
    from utils.general import get_package_version
    tf_version = get_package_version('tensorflow')
    tf_subversion = int(tf_version[1])
    incompatible_tfp_version = (
        '0.99' if (tf_subversion >  16) else
        '0.25' if (tf_subversion == 16) else
        '0.24' if (tf_subversion == 15) else
        '0.23' if (tf_subversion == 14) else
        '0.22' if (tf_subversion == 13) else
        '0.21'
    )
    print(f"Reverting to tensorflow-probability<{incompatible_tfp_version} ...")
    from utils.general import quietly_run
    quietly_run(f'pip install tensorflow-probability<{incompatible_tfp_version}', debug=False)
    import tensorflow_probability as tfp

print("[ √ ] tfp:", tfp.__version__)
tfd = tfp.distributions

import tensorflow as tf  # not before tfp import!
print("[ √ ] tf:", tf.__version__)

# prevent TF from allocating all GPU memory
tf_gpus = tf.config.experimental.list_physical_devices('GPU')
for tf_gpu in tf_gpus:
  tf.config.experimental.set_memory_growth(tf_gpu, True)
from augmentation import adjust_sharpness_tf

π = np.pi
π_half = 0.5 * np.pi
pi = tf.constant(π)
pi_half = tf.constant(π_half)

def init(cfg):
    #cfg.competition_path = Path('/kaggle/input/imagenet-object-localization-challenge')
    #if cfg.cloud == 'drive':
    #    cfg.competition_path = Path(f'/content/gdrive/MyDrive/{cfg.project}')

    if cfg.filetype:
        cfg.image_root = (
            cfg.image_root if (cfg.cloud == 'drive') else
            Path('/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train') if 'imagenet' in cfg.tags else
            Path('/kaggle/input/coco-2017-dataset/coco2017') if 'coco2017' in cfg.tags else
            Path('define image_root in project module!'))
    elif 'imagenet' in cfg.tags or 'imagenetsel' in cfg.tags:
        # Customize data pipeline (see tf_data for definition and defaults)
        # To check out features in new tfrec dataset, set "cfg.tfrec_format = {}"
        """Features in first TFRecord file:
        image/class/synset             bytes_list        0.013 kB
        image/encoded                  bytes_list       32.609 kB
        image/width                    int64_list        0.006 kB
        image/class/label              int64_list        0.006 kB
        image/colorspace               bytes_list        0.007 kB
        image/format                   bytes_list        0.008 kB
        image/height                   int64_list        0.006 kB
        image/channels                 int64_list        0.005 kB
        image/filename                 bytes_list        0.022 kB"""
        cfg.tfrec_format = {'image/encoded': tf.io.FixedLenFeature([], tf.string),
                            'image/height': tf.io.FixedLenFeature([], tf.int64),
                            'image/width': tf.io.FixedLenFeature([], tf.int64)}
        cfg.data_format = {'image': 'image/encoded',
                           'height': 'image/height',
                           'width': 'image/width'}
        cfg.inputs = ['image']

    elif 'coco2017' in cfg.tags:
        """Features in first TFRecord file:
        flickr_url                     bytes_list        0.064 kB
        coco_url                       bytes_list        0.059 kB
        id                             int64_list        0.007 kB
        bboxes                         float_list        0.066 kB
        width                          int64_list        0.006 kB
        area                           float_list        0.020 kB
        objects                        int64_list        0.005 kB
        category_ids                   int64_list        0.008 kB
        annotation_ids                 int64_list        0.016 kB
        date_captured                  bytes_list        0.022 kB
        segmentation_lengths           int64_list        0.009 kB
        image                          bytes_list      219.203 kB
        license                        int64_list        0.005 kB
        file_name                      bytes_list        0.020 kB
        height                         int64_list        0.006 kB
        iscrowd                        int64_list        0.008 kB
        segmentations                  float_list        1.904 kB"""
        cfg.tfrec_format = {'image': tf.io.FixedLenFeature([], tf.string),
                            'height': tf.io.FixedLenFeature([], tf.int64),
                            'width': tf.io.FixedLenFeature([], tf.int64)}
        cfg.data_format = {'image': 'image',
                           'height': 'height',
                           'width': 'width'}
        cfg.inputs = ['image']

    elif 'landmark2021' in cfg.tags:
        cfg.tfrec_format = {'image': tf.io.FixedLenFeature([], tf.string)}
        cfg.data_format = {'image': 'image'}
        cfg.inputs = ['image']

    elif 'landmark2021sel' in cfg.tags:
        cfg.tfrec_format = {'image': tf.io.FixedLenFeature([], tf.string),
                            'target': tf.io.FixedLenFeature([], tf.int64),
                            'species': tf.io.FixedLenFeature([], tf.int64)}
        cfg.data_format = {'image': 'image',
                           'height': 'target',
                           'width': 'species'}
        cfg.inputs = ['image']

    elif 'flickr' in cfg.tags:
        cfg.tfrec_format = {'image': tf.io.FixedLenFeature([], tf.string)}
        cfg.data_format = {'image': 'image'}
        cfg.inputs = ['image']

    elif 'places365' in cfg.tags:
        cfg.tfrec_format = {'image': tf.io.FixedLenFeature([], tf.string)}
        cfg.data_format = {'image': 'image'}
        cfg.inputs = ['image']

    # New datasets for pytorch: Set meta_csv to None to search for images and generate metadata.csv on the fly 
    cfg.meta_csv = (
        cfg.meta_csv if (cfg.cloud == 'drive') else
        Path('/kaggle/input/imagenet-object-localization-challenge/ILSVRC/ImageSets/CLS-LOC/train_cls.txt') if 'imagenet' in cfg.tags else
        Path('/kaggle/input/autolevels-modelbox/coco2017.csv') if 'coco2017' in cfg.tags else
        Path('define cfg.meta_csv in project module!'))

    #elif cfg.filetype == 'tfrec':
    #    cfg.image_root = cfg.competition_path / 'train_tfrecords'

    #cfg.meta_csv = cfg.competition_path / 'train.csv'  # label
    #cfg.gcs_filter = 'train_tfrecords/*.tfrec'
    #if 'tf' in cfg.tags:
    #    cfg.n_classes = 5  # pytorch: set by metadata
    #cfg.gcs_paths = gcs_paths

    if cfg.curve == 'gamma':
        cfg.dataset_class = AugInvGammaDataset
        cfg.channel_size = 3
        cfg.targets = (['target_curve', 'target_bp', 'target_bp2', 'target_log_gamma'] if cfg.output_curve_params else
                       ['target_curve'])
    elif cfg.curve == 'beta':
        cfg.dataset_class = AugInvBetaDataset
        assert cfg.channel_size, "Set channel_size in config file!"
        cfg.targets = (['target_gamma', 'target_bp'] if cfg.channel_size == 6 else
                       ['target_a', 'target_b', 'target_bp'])
    else:
        cfg.dataset_class = FreeCurveDataset
        cfg.channel_size = 256 * 3
        cfg.targets = ['target', 'tfm'] if cfg.catmix else ['target']

    # Preprocessing of model inputs
    if isinstance(cfg.preprocess, str) and ('tf' in cfg.tags):
        # add learnable preprocessing layer (instanciated in models_tf.get_pretrained_model)
        cfg.preprocess = (
            GammaTransformTF if (cfg.preprocess == 'gamma') else
            Curve4TransformTF if (cfg.preprocess == 'curve4') else
            cfg.preprocess)

    elif isinstance(cfg.preprocess, dict):
        # preprocess inputs with static params
        print("Preprocessing images with", cfg.preprocess)
        cfg.preprocess = {key: tf.constant(value) for key, value in cfg.preprocess.items()}
    else:
        print("No preprocessing, cfg.preprocess =", cfg.preprocess)

    # Custom Pooling
    if cfg.pool in {'quantile', 'histogram'}:
        emb_size = (1024 if cfg.arch_name == 'efnv2t' else
                    1280 if cfg.arch_name in [f'efnv2{x}' for x in 's m l xl'.split()] else
                    0 if cfg.arch_name == 'pool_baseline' else None)
        input_shape = [12, 12, emb_size]  # efnv2s, size=384
        bins = cfg.stat_pooling_bins or 64  # image statistic bins per color channel
        add_channels = 1280 if (cfg.arch_name == 'pool_baseline') else 0
        mul_channels = 8 if (cfg.arch_name == 'pool_baseline') else 1

        if cfg.pool == 'histogram':
            cfg.pool = HistogramPooling(input_shape, stat_channels=bins * 3, activation=cfg.act_head,
                                        add_channels=add_channels)
        else:
            cfg.pool = QuantilePooling(input_shape, stat_channels=bins * 3, activation=cfg.act_head,
                                       #add_channels=add_channels,
                                       mul_channels=mul_channels,
                                       name='transform_tf')  # increments body_index by 1


class GammaTransformTF(tf.keras.layers.Layer):
    "Gamma transform with blackpoint shifts before and after, trainable params, for cfg.preprocess"
    def __init__(self):
        super().__init__()
        self.bp = self.add_weight(shape=(3,), name='preprocess/bp', initializer="zeros", trainable=True)
        self.gamma = self.add_weight(shape=(3,), name='preprocess/gamma', initializer="ones", trainable=True)
        self.bp2 = self.add_weight(shape=(3,), name='preprocess/bp2', initializer="zeros", trainable=True)

    def call(self, inputs):
        inputs = tf.clip_by_value(self.bp + inputs * (1 - self.bp), 1e-6, 1)  # avoid nan
        return tf.pow(inputs, self.gamma) * (1 - self.bp2) + self.bp2


class Curve4TransformTF(tf.keras.layers.Layer):
    "Curve4 transform with blackpoint shifts before and after, trainable params, for cfg.preprocess"
    def __init__(self):
        super().__init__()
        self.bp = self.add_weight(shape=(3,), name='preprocess/bp', initializer="zeros", trainable=True)
        self.a = self.add_weight(shape=(3,), name='preprocess/a', trainable=True,
                                 initializer=tf.keras.initializers.Constant([0.5, 0.5, 0.5]))
        self.b = self.add_weight(shape=(3,), name='preprocess/b', trainable=True,
                                 initializer=tf.keras.initializers.Constant([8/(π**2), 8/(π**2), 8/(π**2)]))
        self.bp2 = self.add_weight(shape=(3,), name='preprocess/bp2', initializer="zeros", trainable=True)

    def call(self, inputs):
        inputs = tf.clip_by_value(self.bp + inputs * (1 - self.bp), 1e-6, 1 - 1e-6)  # avoid nan
        return (1 - tf.cos(pi_half * inputs ** self.a) ** self.b) * (1 - self.bp2) + self.bp2


def find_images(cfg):
    "Recursively find all images and return them in a sorted list"

    image_root = Path(cfg.image_root)
    filetype = cfg.filetype
    if cfg.DEBUG:
        print(f"Searching recursively for {filetype} images in {image_root}")

    fns = sorted(p.relative_to(image_root).as_posix() for p in image_root.glob(f'**/*.{filetype}'))

    if cfg.DEBUG:
        print(f"    found {len(fns)} images")

    return fns



def read_csv(cfg):
    "Return pandas DataFrame with image_id and image_path, relative to cfg.image_root"

    if cfg.meta_csv is None:
        image_paths = find_images(cfg)
        image_ids = [Path(s).stem for s in image_paths]
        df = pd.DataFrame({'image_id': image_ids, 'image_path': image_paths})
        df.to_csv('metadata.csv', index=False)
    elif 'imagenet' in cfg.tags:
        df = pd.read_csv(cfg.meta_csv, sep=' ', usecols=[0], header=None, names=['image_id'])
    else:
        df = pd.read_csv(cfg.meta_csv)

    return df.sample(frac=0.01) if cfg.DEBUG else df


def adjust_sharpness_alb(img, sharpness):
    """Equivalent to torchvision.transforms.functional.adjust_sharpness
    
    kernel differs from torchvision: sharpness=1.7 corresponds to torchvision sharpness=2.0
    """
    img = img.astype(np.float32) * sharpness + blur(img, ksize=3).astype(np.float32) * (1 - sharpness)
    return img.clip(0, 255).astype(np.uint8)


def adjust_jpeg_quality_alb(img, quality):
    return ImageCompression(quality_lower=quality, quality_upper=quality, 
                            always_apply=True)(image=img)['image']


def adjust_jpeg_quality_tvf(img, quality):
    "Returns HWC numpy image or (return_jpeg=True) CHW tensor"
    img = torch.tensor(img).permute(2, 0, 1)
    jpeg = encode_jpeg(img, quality)
    return decode_jpeg(jpeg).permute(1, 2, 0).numpy()


def map_index_torch_old(image, tfm, add_uniform_noise=False, resize=None):
    # map image (H, W, C) to tfm (C, 256) using torch.gather (faster than numpy fancy indexing)
    # tfm must be expanded to have same shape as image (except dim=1)
    # Returns HWC uint8 numpy 
    expanded_curves = torch.tensor(tfm.T)[None, :, :].expand(image.shape[0], -1, -1)
    if not add_uniform_noise:
        image = torch.gather(expanded_curves, dim=1, index=torch.LongTensor(image))  # HWC float
        if resize is None:
            return (image.numpy().clip(0, 1) * 255).astype(np.uint8)  # HWC uint8 numpy
        else:            
            image = resize((image.permute(2, 0, 1) * 255).to(torch.uint8))  # CHW uint8
            return image.permute(1, 2, 0).numpy()  # HWC uint8 numpy

    # add uniform noise to mask uint8 quantization (slow)
    image_plus_one = torch.gather(expanded_curves, dim=1, index=(torch.LongTensor(image) + 1).clamp(max=255))
    image = torch.gather(expanded_curves, dim=1, index=torch.LongTensor(image))
    image = image + torch.rand_like(image) * (image_plus_one - image)  # HWC float
    return (image.numpy().clip(0, 1) * 255).astype(np.uint8)  # HWC uint8 numpy


def map_index_torch(image, tfm, add_uniform_noise=False, resize=None):
    # map image (H, W, C) to tfm (C, 256) using torch.gather (faster than numpy fancy indexing)
    # tfm must be expanded to have same shape as image (except dim=1)
    # Returns HWC uint8 numpy
    tfm = torch.tensor(tfm) if not isinstance(tfm, torch.Tensor) else tfm
    tfm = (tfm.clamp(0, 1) * 255).to(torch.uint8)
    expanded_curves = tfm.T[None, :, :].expand(image.shape[0], -1, -1)
    if not add_uniform_noise:
        image = torch.gather(expanded_curves, dim=1, index=torch.LongTensor(image))  # HWC float
        if resize is None:
            return image.numpy()  # HWC uint8 numpy
        else:            
            image = resize(image.permute(2, 0, 1))  # CHW uint8
            return image.permute(1, 2, 0).numpy()  # HWC uint8 numpy

    # add uniform noise to mask uint8 quantization (slow)
    image_plus_one = torch.gather(expanded_curves, dim=1, index=(torch.LongTensor(image) + 1).clamp(max=255))
    image = torch.gather(expanded_curves, dim=1, index=torch.LongTensor(image))
    image = image + torch.rand_like(image) * (image_plus_one - image)  # HWC float
    return (image.numpy().clip(0, 1) * 255).astype(np.uint8)  # HWC uint8 numpy


def get_pretrained_model(cfg, strategy):
    if cfg.arch_name == 'pool_baseline':
        return get_pool_baseline_model(cfg, strategy)
    if cfg.arch_name.startswith('stat_augmented_efnv2'):
        return get_stat_augmented_efnv2_model(cfg, strategy)
    else:
        from models_tf import get_pretrained_model as default_get_pretrained_model
        return default_get_pretrained_model(cfg, strategy)


def get_stat_augmented_efnv2_model(cfg, strategy):
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense, Dropout
    from normalization import Normalization
    from models_tf import get_bottleneck_params
    import keras_efficientnet_v2 as efn

    with strategy.scope():

        # Inputs & Pooling
        input_shape = (*cfg.size, 3)
        inputs = [Input(shape=input_shape, name='image')]

        # Body and Feature Extractor
        model_cls = getattr(efn, f'EfficientNetV2{cfg.arch_name[20:].upper()}')
        pretrained_model = model_cls(input_shape=input_shape, num_classes=0, pretrained=cfg.pretrained)

        features = pretrained_model(inputs[0])
        if not isinstance(features, list):
            # Currently, there is no feature extraction option for neither
            # keras_efficientnet_v2 nor tf.keras.applications.EfficientNetV2* classes
            raise NotImplementedError('implement feature extractor: body(x) --> features')
        pooled_features = [cfg.pool(x, inputs[0]) for x in features]
        embed = tf.concat(pooled_features, axis=-1)

        # Bottleneck
        lin_ftrs, dropout_ps, final_dropout = get_bottleneck_params(cfg)
        for i, (p, out_channels) in enumerate(zip(dropout_ps, lin_ftrs)):
            embed = Dropout(p, name=f"dropout_{i}_{p}")(embed) if p > 0 else embed
            embed = Dense(out_channels, activation=cfg.act_head, name=f"FC_{i}")(embed)
            embed = Normalization(cfg.normalization_head, name=f"BN_{i}")(embed) if cfg.normalization_head else embed
        embed = Dropout(final_dropout, name=f"dropout_final_{final_dropout}")(
            embed) if final_dropout else embed
        if cfg.normalization_head and not lin_ftrs:
            embed = Normalization(cfg.normalization_head, name="BN_final")(embed)  # does this help?

        # Output
        assert cfg.curve and (cfg.curve == 'free')
        output = Dense(cfg.channel_size, name='regressor')(embed)
        outputs = [output]

        # Build model
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        assert not (cfg.rst_path and cfg.rst_name)

        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr, beta_1=cfg.betas[0], beta_2=cfg.betas[1])
        metrics_classes = {'curve_rmse': TFCurveRMSE(curve=cfg.curve)}
        metrics = [metrics_classes[m] for m in cfg.metrics]
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=metrics)

    return model


def get_pool_baseline_model(cfg, strategy):
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense, Dropout
    from normalization import Normalization
    from models_tf import get_bottleneck_params, set_trainable, peek_layer_weights

    with strategy.scope():

        # Inputs & Pooling
        input_shape = (*cfg.size, 3)
        inputs = [Input(shape=input_shape, name='image')]
        x = None
        embed = cfg.pool(x, inputs)

        # Bottleneck
        lin_ftrs, dropout_ps, final_dropout = get_bottleneck_params(cfg)
        for i, (p, out_channels) in enumerate(zip(dropout_ps, lin_ftrs)):
            embed = Dropout(p, name=f"dropout_{i}_{p}")(embed) if p > 0 else embed
            embed = Dense(out_channels, activation=cfg.act_head, name=f"FC_{i}")(embed)
            embed = Normalization(cfg.normalization_head, name=f"BN_{i}")(embed) if cfg.normalization_head else embed
        embed = Dropout(final_dropout, name=f"dropout_final_{final_dropout}")(
            embed) if final_dropout else embed
        if cfg.normalization_head and not lin_ftrs:
            embed = Normalization(cfg.normalization_head, name="BN_final")(embed)  # does this help?

        # Output
        assert cfg.curve and (cfg.curve == 'free')
        output = Dense(cfg.channel_size, name='regressor')(embed)
        outputs = [output]

        # Build model
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        #peek_layer_weights(model)

        # Load restart weights
        if cfg.rst_path and cfg.rst_name:
            # avoid ValueError "axes don't match array":
            # https://stackoverflow.com/questions/51944836/keras-load-model-valueerror-axes-dont-match-array
            set_trainable(model, cfg.freeze_for_loading)

            try:
                model.load_weights(Path(cfg.rst_path) / cfg.rst_name)
            except ValueError:
                print(f"{cfg.rst_name} mismatches model with body: {model.layers[1].name}")
                print("Trying to load matching layers only...")
                model.load_weights(Path(cfg.rst_path) / cfg.rst_name, 
                                   by_name=True, skip_mismatch=True)
            print(f"Weights loaded from {cfg.rst_name}")

        #peek_layer_weights(model)
        #for w in model.layers[1].weights:
        #    print("    " + f"{w.name}: {w.shape}")

        # Freeze/unfreeze, set BN parameters
        set_trainable(model, cfg.freeze)

        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr, beta_1=cfg.betas[0], beta_2=cfg.betas[1])
        metrics_classes = {'curve_rmse': TFCurveRMSE(curve=cfg.curve)}
        metrics = [metrics_classes[m] for m in cfg.metrics]
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=metrics)

    return model


class Curve0():
    def __init__(self, gamma=1.0, bp=0, bp2=0, bp_clip=None, unclipped=False):
        """Function  y(x) = x^a  with offsets bp, bp2 in x, y

        Input/Output range: [0, 1]

        Parameters
        gamma: min, mean, max [0.25, 1, 3]
        bp:  8-bit offset in x, range [-inf, 255]
        bp2: 8-bit offset in y, range [-inf, 255]

        inverse() returns the inverse function 
            y^-1(x) = x^(1 / gamma)
        """
        self.params = [np.array(p, dtype=np.float32) for p in (gamma, bp / 255, bp2 / 255)]
        self.x_min = 1e-6
        self.x_max = None
        self.bp_clip = max(int(bp_clip), 0) if bp_clip is not None else None
        self.bp_is_clipped = False
        self.unclipped = bool(unclipped)

    def __call__(self, x):
        assert (x.shape[0] in {1, 3}) and (x.ndim > 1), f'x has wrong shape: {x.shape}, expecting (C, *)'
        if not self.bp_is_clipped: _ = self.inverse(x)
        gamma, bp, bp2 = (p[:, None] for p in self.params)

        x = bp + x * (1 - bp)
        x = np.clip(x, self.x_min, self.x_max)  # avoid nan
        x = np.power(x, gamma)
        x = x * (1 - bp2) + bp2
        return x if self.unclipped else np.clip(x, 0, 1)
    
    def inverse(self, x):
        assert (x.shape[0] in {1, 3}) and (x.ndim > 1), f'x has wrong shape: {x.shape}, expecting (C, *)'
        gamma, bp, bp2 = (p[:, None] for p in self.params)

        x = (x - bp2) / (1 - bp2)
        x = np.clip(x, self.x_min, self.x_max)  # avoid nan
        x = np.power(x, 1 / gamma)
        if self.bp_clip:
            assert x.shape == (3, 256), f'x has unexpected shape {x.shape}'
            bp_max = x[:, self.bp_clip:self.bp_clip + 1]
            bp = np.minimum(bp, bp_max)
            self.params[1] = bp[:, 0]  # propagate to future calls
            self.bp_is_clipped = True
        x = (x - bp) / (1 - bp)
        return x if self.unclipped else np.clip(x, 0, 1)


class Curve3():
    def __init__(self, alpha=2.0, beta=0.99, bp=0, bp2=0, bp2_clip=None, mirror_mask=None, unclipped=False):
        """Function y(x) = Beta(alpha, beta).PDF(x)  with offsets bp, bp2 in x, y

        Input/Output range: [0, 1]

        Parameters
        alpha: min, mean, max [1.4, 2.0, 2.9]
        beta:  min, mean, max [0.5, 0.75, 1.0]
        bp:    8-bit offset in x, range [-inf, 255]
        bp2:   8-bit offset in y, range [-inf, 255]

        inverse() returns the inverse function
        """
        assert np.min(alpha) > 1, f'alpha ({alpha}) out of scope, must be > 1'
        assert (np.min(beta) > 0) and (np.max(beta) <= 1), f'beta ({beta}) out of scope, must be in (0, 1]'
        self.params = [np.array(p, dtype=np.float32) for p in (alpha, beta, bp / 255, bp2 / 255)]
        self.x_min = 1e-6
        self.x_max = 1.0 - 1e-6
        self.eps = 0.1
        self.bp2_clip = max(int(bp2_clip), 0) if bp2_clip is not None else None
        self.bp2_is_clipped = False
        self.mirror_mask = np.ones((3, 1), dtype=np.float32) if mirror_mask is None else mirror_mask
        self.unclipped = bool(unclipped)

    def __call__(self, x):
        assert (x.shape[0] in {1, 3}) and (x.ndim > 1), f'x has wrong shape: {x.shape}, expecting (C, *)'
        alpha, beta, bp, bp2 = (p[:, None] for p in self.params)

        x = bp + x * (1 - bp)
        x = self.mirror_mask * self.pdf(x, alpha, beta) + (1 - self.mirror_mask) * self.inverse_pdf(x, alpha, beta)
        if self.bp2_clip is not None and self.bp2_clip < 255:
            bp2_max = (self.bp2_clip / 255 - x[:, 0:1]) / (1 - x[:, 0:1])
            bp2 = np.minimum(bp2, bp2_max)
            self.params[3] = bp2[:, 0]  # propagate to future calls
            self.bp2_is_clipped = True
        x = x * (1 - bp2) + bp2
        return x if self.unclipped else np.clip(x, 0, 1)
    
    def inverse(self, x):
        assert (x.shape[0] in {1, 3}) and (x.ndim > 1), f'x has wrong shape: {x.shape}, expecting (C, *)'
        if self.bp2_clip is not None and self.bp2_is_clipped is False: _ = self(x)
        alpha, beta, bp, bp2 = (p[:, None] for p in self.params)
        
        x = (x - bp2) / (1 - bp2)
        x = self.mirror_mask * self.inverse_pdf(x, alpha, beta) + (1 - self.mirror_mask) * self.pdf(x, alpha, beta)
        x = (x - bp) / (1 - bp)
        return x if self.unclipped else np.clip(x, 0, 1)

    def pdf(self, x, alpha, beta):
        x = np.clip(x, self.x_min, self.x_max)  # avoid nan
        x = x * (1 - self.eps)                  # reduce slope at whitepoint
        x = np.power(x, alpha - 1) * np.power(1 - x, beta - 1)  # unnormalized PDF(x)
        x /= x[:, -1:]                                          # normalize
        return x
    
    def inverse_pdf(self, xs, alpha, beta):
        y = np.linspace(self.x_min, self.x_max, 2000, dtype=np.float64)  # float64 avoids div-by-zero below
        pdfs = self.pdf(y[None, :], alpha, beta)
        assert pdfs.shape[0] == 3, str(pdfs.shape)
        xs = xs.repeat(3, axis=0) if xs.shape[0] == 1 else xs
        # fill_value='extrapolate' produces NaNs
        return np.stack([interp1d(pdf, y, fill_value=(x[0], x[-1]), bounds_error=False,
                                  assume_sorted=True)(x).clip(0, 1).astype(xs.dtype) for x, pdf in zip(xs, pdfs)])


class Curve4():
    def __init__(self, a=0.5, b=0.81, bp=0, bp2=0, bp_clip=None, mirror_mask=None, unclipped=False):
        """Function y(x) = 1 - cos(π/2 * x^a)^b  with offsets bp, bp2 in x, y

        Input/Output range: [0, 1]

        Parameters
        a:   min, mean, max [0.2, 0.5, 2]
        b:   min, mean, max [0.5, 8/(π**2), 2]
        bp:  8-bit offset in x, range [-inf, 255]
        bp2: 8-bit offset in y, range [-inf, 255]

        inverse() returns the inverse function 
            y^-1(x) = (2 / π)**(1 / a) * np.arccos((1 - x)**(1 / b))**(1 / a)
        """
        self.params = [np.array(p, dtype=np.float32) for p in (a, b, bp / 255, bp2 / 255)]
        self.x_min = 1e-6
        self.x_max = 1.0 - 1e-6
        self.bp_clip = max(int(bp_clip), 0) if bp_clip is not None else None
        self.bp_is_clipped = False
        self.mirror_mask = np.ones((3, 1), dtype=np.float32) if mirror_mask is None else mirror_mask
        self.unclipped = bool(unclipped)

    def __call__(self, x):
        assert (x.shape[0] in {1, 3}) and (x.ndim > 1), f'x has wrong shape: {x.shape}, expecting (C, *)'
        if not self.bp_is_clipped: _ = self.inverse(x)
        a, b, bp, bp2 = (p[:, None] for p in self.params)

        x = bp + x * (1 - bp)
        x = self.mirror_mask * self.curve4(x, a, b) + (1 - self.mirror_mask) * self.inv_curve4(x, a, b)
        x = x * (1 - bp2) + bp2
        return x if self.unclipped else np.clip(x, 0, 1)
    
    def inverse(self, x):
        assert (x.shape[0] in {1, 3}) and (x.ndim > 1), f'x has wrong shape: {x.shape}, expecting (C, *)'
        a, b, bp, bp2 = (p[:, None] for p in self.params)

        x = (x - bp2) / (1 - bp2)
        x = self.mirror_mask * self.inv_curve4(x, a, b) + (1 - self.mirror_mask) * self.curve4(x,  a, b)
        if self.bp_clip:
            assert x.shape == (3, 256), f'x has unexpected shape {x.shape}'
            bp_max = x[:, self.bp_clip:self.bp_clip + 1]
            bp = np.minimum(bp, bp_max)
            self.params[2] = bp[:, 0]  # propagate to future calls
            self.bp_is_clipped = True
        x = (x - bp) / (1 - bp)
        return x if self.unclipped else np.clip(x, 0, 1)

    def curve4(self, x, a, b):
        x = np.clip(x, self.x_min, self.x_max)  # avoid nan
        return 1 - np.cos(π_half * x ** a) ** b

    def inv_curve4(self, x, a, b):
        x = np.clip(x, self.x_min, self.x_max)  # avoid nan
        return π_half**(-1 / a) * np.arccos((1 - x)**(1 / b))**(1 / a)


class AugInvGammaDataset(Dataset):
    """Images are transformed according to randomly drawn curve parameters
    
    Floatify, inv-curve-transform, noise, blur, uint8, crop/resize, tensorize"""

    def __init__(self, df, cfg, labeled=True, transform=None, tensor_transform=None,
                 return_path_attr=None):
        """
        Args:
            df (pd.DataFrame):                First row must contain the image file paths
            image_root (string, Path):        Root directory for df.image_path
            transform (callable, optional):   Optional transform to be applied on the first
                                              element (image) of a sample.
            labeled (bool, optional):         if True, return curve parameters as regression target
            return_path_attr (str, optional): return Path attribute `return_path_attr`

        """
        self.df = df.reset_index(drop=True)
        self.image_root = cfg.image_root
        self.transform = transform
        self.tensor_transform = tensor_transform
        self.albu = transform and transform.__module__.startswith('albumentations')
        self.floatify = not (self.albu and 'Normalize' in [t.__class__.__name__ for t in transform])
        self.labeled = labeled
        self.return_path_attr = return_path_attr
        self.use_batch_tfms = cfg.use_batch_tfms
        if self.use_batch_tfms:
            self.presize = TT.Resize([int(s * cfg.presize) for s in cfg.size], interpolation=InterpolationMode.NEAREST)
        #self.dist_log_gamma = torch.distributions.normal.Normal(0, 0.4)
        self.dist_log_gamma = torch.distributions.uniform.Uniform(-1.6, 1.0)
        self.noise_level = cfg.noise_level
        self.dist_bp = torch.distributions.normal.Normal(0, cfg.random_blackpoint_shift / 255) if cfg.random_blackpoint_shift else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        fn = os.path.join(self.image_root, self.df.iloc[index, 0])
        if 'gcsfs' in globals() and gcsfs.is_gcs_path(fn):
            bytes_data = gcsfs.read(fn)
            image = PIL.Image.open(io.BytesIO(bytes_data))
        else:
            assert os.path.exists(fn), f'{fn} not found'
            image = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)

        n_channels = image.shape[2]
        assert n_channels in {1, 3}, f'wrong image shape: {image.shape}, expecting channel last'

        # draw gamma, create labels
        abs_log_gamma = self.dist_log_gamma.sample((n_channels,))
        gamma = torch.exp(abs_log_gamma)
        rel_log_gamma = abs_log_gamma - torch.mean(abs_log_gamma)
        labels = torch.cat((gamma, abs_log_gamma, rel_log_gamma))

        if self.use_batch_tfms:
            # append rnd_factor for noise_level
            if self.noise_level:
                rnd_factor = torch.rand(1)
                labels = torch.cat((labels, rnd_factor))

            # append random blackpoint shift
            if self.dist_bp:
                rnd_bp_shift = self.dist_bp.sample((n_channels,))
                labels = torch.cat((labels, rnd_bp_shift))

            # resize image to double cfg.size and tensorize for collocation
            image = torch.tensor(image.transpose(2, 0, 1))  # channel first
            image = self.presize(image)

            return image, labels


        # this is slow on CPU, use batch tfms to do it on TPU
        image = np.array(image, dtype=np.float32) / 255
        image = np.power(image, labels[None, None, :].numpy())  # channel last
        if self.noise_level:
            image += np.random.randn(*image.shape) * self.noise_level
        image *= 255
        image = np.clip(image, 0, 255).astype(np.uint8)

        if self.transform:
            if self.albu:
                image = self.transform(image=np.array(image))['image']
                image = (image / 255).float() if self.floatify else image
            else:
                # torchvision, requires PIL.Image
                image = self.transform(PIL.Image.fromarray(image))

        if self.tensor_transform:
            image = self.tensor_transform(image)

        return image, labels if self.labeled else image


class AugInvBetaDataset(Dataset):
    """Images are transformed according to randomly drawn curve parameters
    
    Floatify, inv-curve-transform, noise, blur, uint8, crop/resize, tensorize"""

    def __init__(self, df, cfg, labeled=True, transform=None, tensor_transform=None,
                 return_path_attr=None):
        """
        Args:
            df (pd.DataFrame):                First row must contain the image file paths
            image_root (string, Path):        Root directory for df.image_path
            transform (callable, optional):   Optional transform to be applied on the first
                                              element (image) of a sample.
            labeled (bool, optional):         if True, return curve parameters as regression target
            return_path_attr (str, optional): return Path attribute `return_path_attr`

        """
        self.df = df.reset_index(drop=True)
        self.image_root = cfg.image_root
        self.transform = transform
        self.tensor_transform = tensor_transform
        self.albu = transform and transform.__module__.startswith('albumentations')
        self.floatify = not (self.albu and 'Normalize' in [t.__class__.__name__ for t in transform])
        self.labeled = labeled
        self.return_path_attr = return_path_attr
        self.dist_a = torch.distributions.normal.Normal(0, cfg.a_sigma or 0.5)
        self.dist_b = torch.distributions.normal.Normal(cfg.b_mean or 0.4, cfg.b_sigma or 0.25)
        self.dist_bp = torch.distributions.half_normal.HalfNormal(cfg.bp_sigma or 0.02)
        self.alpha_scale = cfg.alpha_scale or 1
        self.beta_decay = cfg.beta_decay or 10
        self.use_batch_tfms = cfg.use_batch_tfms
        if self.use_batch_tfms:
            self.presize = TT.Resize([int(s * cfg.presize) for s in cfg.size], interpolation=InterpolationMode.NEAREST)
        self.noise_level = cfg.noise_level
        self.curve_tfm_on_device = cfg.curve_tfm_on_device

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        fn = os.path.join(self.image_root, self.df.iloc[index, 0])
        if 'gcsfs' in globals() and gcsfs.is_gcs_path(fn):
            bytes_data = gcsfs.read(fn)
            image = PIL.Image.open(io.BytesIO(bytes_data))
        else:
            assert os.path.exists(fn), f'{fn} not found'
            image = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)

        n_channels = image.shape[-1]
        assert n_channels in {1, 3}, f'wrong image shape: {image.shape}, expecting channel last'
        
        labels = [dist.sample((n_channels,)) for dist in [self.dist_a, self.dist_b, self.dist_bp]]
        curves = Curve(*tuple(p.numpy() for p in labels), create_blackpoint=True, 
                       alpha_scale=self.alpha_scale, beta_decay=self.beta_decay)
        labels = torch.stack(labels, axis=1)  # channel first
        #assert labels.shape == (n_channels, 3)

        if self.use_batch_tfms:
            # just resize to double cfg.size and tensorize for collocation
            if self.curve_tfm_on_device:
                image = torch.tensor(image.transpose(2, 0, 1))  # channel first
                image = self.presize(image)
                curves = torch.tensor(curves.get_inverse_curves(), dtype=torch.float32)
                #assert curves.shape == (n_channels, 256)
                #assert curves.dtype == labels.dtype, f'{curves.dtype} != {labels.dtype}'
                return image, torch.cat([labels, curves], axis=1)
            else:
                image = curves.apply_inverse_pdf(image)
                image = torch.tensor(image.transpose(2, 0, 1))  # channel first
                image = self.presize(image)
                return image, labels

        image = curves.apply_inverse_pdf(image)

        if self.transform:
            if self.albu:
                image = self.transform(image=np.array(image))['image']
                image = (image / 255).float() if self.floatify else image
            else:
                # torchvision, requires PIL.Image
                image = self.transform(PIL.Image.fromarray(image))

        if self.tensor_transform:
            image = self.tensor_transform(image)

        return image, labels if self.labeled else image


def get_mask(ps, n_channels=3, n_samples=10):
    """Returns float32 mask with shape [n_channels, len(ps)]
    
    Elements are in range 0...1 and sum up to 1 over axis 1.
    n_samples: number of draws to be averaged for each channel"""
    sample = np.random.multinomial(1, ps, size=(n_channels, n_samples))
    return np.mean(sample, axis=1, dtype=np.float32)


class FreeCurveDataset(Dataset):
    """Images are mapped on device using the channelwise-randomly generated target_curve"""

    def __init__(self, df, cfg, labeled=True, transform=None, tensor_transform=None,
                 return_path_attr=None):
        """
        Args:
            df (pd.DataFrame):                First row must contain the image file paths
            image_root (string, Path):        Root directory for df.image_path
            transform (callable, optional):   Optional transform to be applied on the first
                                              element (image) of a sample.
            labeled (bool, optional):         if True, return target_curve
            return_path_attr (str, optional): return Path attribute `return_path_attr`

        """
        self.df = df.reset_index(drop=True)
        self.image_root = cfg.image_root
        self.transform = transform
        self.tensor_transform = tensor_transform
        self.albu = transform and transform.__module__.startswith('albumentations')
        self.floatify = not (self.albu and 'Normalize' in [t.__class__.__name__ for t in transform])
        self.labeled = labeled
        self.return_path_attr = return_path_attr
        self.use_batch_tfms = cfg.use_batch_tfms
        self.resize_before_jpeg = cfg.resize_before_jpeg
        if self.use_batch_tfms:
            self.presize = TT.Resize([int(s * cfg.presize) for s in cfg.size],
                                     interpolation=InterpolationMode.NEAREST, antialias=cfg.antialias)
        else:
            interpolation = getattr(InterpolationMode, cfg.interpolation.upper() if cfg.interpolation else 'NEAREST')
            self.resize = TT.Resize(cfg.size, interpolation=interpolation, antialias=cfg.antialias)

        self.log_gamma_range = cfg.log_gamma_range
        self.curve3_a_range = cfg.curve3_a_range
        self.curve3_beta_range = cfg.curve3_beta_range
        self.curve4_loga_range = cfg.curve4_loga_range
        self.curve4_b_range = cfg.curve4_b_range
        self.bp_range = cfg.blackpoint_range
        self.bp2_range = cfg.blackpoint2_range
        self.bp_clip = max(*cfg.blackpoint_range, *cfg.blackpoint2_range) if cfg.clip_target_blackpoint else None
        self.p_gamma = cfg.p_gamma  # probability to use gamma (Curve0)
        self.p_beta = cfg.p_beta    # probability for Beta PDF (Curve3) rather than Curve4
        self.noise_level = cfg.noise_level
        self.add_uniform_noise = cfg.add_uniform_noise
        self.add_jpeg_artifacts = cfg.add_jpeg_artifacts
        self.sharpness_augment = cfg.sharpness_augment
        self.predict_inverse = cfg.predict_inverse
        self.mirror_beta = cfg.mirror_beta
        self.mirror_curve4 = cfg.mirror_curve4
        self.curve_selection = cfg.curve_selection or 'channel-wise'  # 'channel-wise' or 'image-wise'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        fn = os.path.join(self.image_root, self.df.iloc[index, 0])
        if 'gcsfs' in globals() and gcsfs.is_gcs_path(fn):
            bytes_data = gcsfs.read(fn)
            image = PIL.Image.open(io.BytesIO(bytes_data))
        else:
            assert os.path.exists(fn), f'{fn} not found'
            image = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)

        n_channels = image.shape[2]
        assert n_channels in {1, 3}, f'wrong image shape: {image.shape}, expecting channel last'

        # Generate curve (C, 256)
        support = np.linspace(0, 1, 256, dtype=np.float32)
        bp = np.random.uniform(*self.bp_range, n_channels).astype(np.float32)
        bp2 = np.random.uniform(*self.bp2_range, n_channels).astype(np.float32)
        curves = []
        
        # gamma
        log_gamma = np.random.uniform(*self.log_gamma_range, n_channels).astype(np.float32)
        gamma = np.exp(log_gamma)
        curves.append(Curve0(gamma, bp, bp2, self.bp_clip))

        # beta
        alpha = np.exp(np.random.uniform(*self.curve3_a_range, n_channels).astype(np.float32))
        beta = np.random.uniform(*self.curve3_beta_range, n_channels).astype(np.float32)
        mirror_mask = np.random.randint(low=0, high=2, size=(3, 1)).astype(np.float32) if self.mirror_beta else None
        curves.append(Curve3(alpha, beta, bp, bp2, self.bp_clip, mirror_mask))

        # curve4
        a = np.exp(np.random.uniform(*self.curve4_loga_range, n_channels).astype(np.float32))
        b = np.random.uniform(*self.curve4_b_range, n_channels).astype(np.float32)
        mirror_mask = np.random.randint(low=0, high=2, size=(3, 1)).astype(np.float32) if self.mirror_curve4 else None
        curves.append(Curve4(a, b, bp, bp2, self.bp_clip, mirror_mask))

        # RNG DEBUG: checked: different random numbers on each torchrun instance.
        #print(f"bp={bp[0]:4.1f} bp2={bp2[2]:4.1f} g={gamma[0]:4.2f} alpha={alpha[2]:4.2f} beta={beta[0]:4.2f} a={a[2]:4.2f} b={b[0]:4.2f}")

        if self.curve_selection == 'channel-wise':
            targets, tfms = [], []
            for curve in curves:
                tfm = curve.inverse(support[None, :])  # shape (n_channels, 256)
                target = curve(support[None, :]) if self.predict_inverse else tfm
                targets.append(target)
                tfms.append(tfm)
            targets = np.stack(targets)  # shape (n_curves, n_channels, 256)
            tfms = np.stack(tfms)
            
            p_gamma = self.p_gamma
            p_beta = (1 - self.p_gamma) * self.p_beta
            p_curve4 = (1 - p_gamma - p_beta)
            mask = get_mask([p_gamma, p_beta, p_curve4], n_samples=1)
            target = np.einsum('ji,ijk->jk', mask, targets)
            tfm = np.einsum('ji,ijk->jk', mask, tfms)
            del targets, tfms
        else:
            # image-wise curve selection
            curve = curves[np.random.randint(0, 3)]
            tfm = curve.inverse(support[None, :])
            target = curve(support[None, :]) if self.predict_inverse else tfm

            # random swap tfm <-> target
            if (self.predict_inverse and (np.random.random_sample() < 0.5) and any([
                (curve.__class__.__name__ == 'Curve3') and self.mirror_beta,
                (curve.__class__.__name__ == 'Curve4') and self.mirror_curve4])):
                mask = np.random.randint(0, 2, (n_channels, 1)).astype(np.float32)
                target, tfm = mask * target + (1 - mask) * tfm, (1 - mask) * target + mask * tfm
        
        target = torch.tensor(target)
        tfm = torch.tensor(tfm)
        assert target.shape == (n_channels, 256), f"wrong target shape: {target.shape}"
        assert target.dtype == torch.float32, f"wrong target dtype: {target.dtype}"

        if self.use_batch_tfms:
            # return both target and tfm curves as "target"
            if self.predict_inverse and self.noise_level:
                # append rnd_factor for noise_level to target, independent of use_batch_tfms
                rnd_factor = torch.rand(1).repeat(3)
                target = torch.cat((target, rnd_factor[:, None], tfm), dim=1)
            elif self.predict_inverse:
                target = torch.cat((target, tfm), dim=1)

            # append rnd JPEG quality -> (C, 258)
            if self.add_jpeg_artifacts:
                jpeg_quality = torch.randint(50, 100, (1,)).repeat(3).float()
                target = torch.cat((target, jpeg_quality[:, None]), dim=1)

            # append rnd sharpness -> (C, 259)
            if self.sharpness_augment:
                rnd_sharpness = 2 * torch.rand((1,)).repeat(3)
                target = torch.cat((target, rnd_sharpness[:, None]), dim=1)

            # resize image to double cfg.size and tensorize for collocation
            image = torch.tensor(image.transpose(2, 0, 1))  # channel first
            image = self.presize(image)

            if self.return_path_attr:
                path_attr = (
                    Path(fn).relative_to(self.image_root).as_posix() if self.return_path_attr.startswith('relative') else
                    getattr(Path(fn), self.return_path_attr))
                return image, target, path_attr

            return image, target

        resize = self.resize if self.resize_before_jpeg else None
        if max(image.shape) < max(self.resize.size):
            resize = None  # resize only large images
        #print("image before map_index:", image.shape, type(image), image.dtype, image.max())
        #print("tfm before map_index:", tfm.shape, type(tfm), tfm.dtype, tfm.max())
        image = map_index_torch(image, tfm, self.add_uniform_noise, resize)
        #print("image after map_index:", image.shape, type(image), image.dtype, image.max())

        if self.sharpness_augment:
            # randomly soften/sharpen the image
            rnd_sharpness = 1.8 * np.random.rand() + 0.1
            image = adjust_sharpness_alb(image, rnd_sharpness)

        if self.add_jpeg_artifacts:
            # adjust_jpeg_quality automatically converts image to uint8 and back
            rnd_quality = int(50 * (1 + np.random.rand()))
            resize = None
            image = adjust_jpeg_quality_tvf(image, rnd_quality)

        image = image.astype(np.float32) / 255

        # append rnd_factor for noise_level to target -> (C, 257)
        if self.noise_level:
            rnd_factor = torch.rand(1).repeat(3)
            target = torch.cat((target, rnd_factor[:, None]), dim=1)

        image = torch.tensor(image).permute(2, 0, 1)

        if not self.resize_before_jpeg or any(a != b for a, b in zip(image.shape[1:], self.resize.size)):
            image = self.resize(image)

        if self.return_path_attr:
            path_attr = (
                Path(fn).relative_to(self.image_root).as_posix() if self.return_path_attr.startswith('relative') else
                getattr(Path(fn), self.return_path_attr))
            return image, target, path_attr

        return image, target


class StatPooling(tf.keras.layers.Layer):
    requires_inputs = True  # if this class attribute exists: call(embed, inputs) 
    def __init__(self, input_shape, stat_channels, mul_channels=1, add_channels=0, squeeze=False, activation=None, **kwargs):
        """
        input_shape: embed shape, e.g., (12, 12, 1280)
        channels:  channels of the image statistics, max: 256 * 3
        """
        super().__init__(**kwargs)
        self.fields = input_shape[:2]
        emb_ch = input_shape[2]
        self.stat_channels = stat_channels
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.bn6 = tf.keras.layers.BatchNormalization()
        self.bn7 = tf.keras.layers.BatchNormalization()
        self.bn8 = tf.keras.layers.BatchNormalization()
        #self.bn9 = tf.keras.layers.BatchNormalization()
        #self.bn10 = tf.keras.layers.BatchNormalization()
        #self.bn11 = tf.keras.layers.BatchNormalization()
        #self.bn12 = tf.keras.layers.BatchNormalization()
        c1 = stat_channels * 8
        c2 = (stat_channels + emb_ch) * mul_channels + add_channels
        #c1 = c2  # merge-4-skipmerge-4
        self.conv1 = tf.keras.layers.Conv2D(c1, 1, activation=activation)
        self.conv2 = tf.keras.layers.Conv2D(c1, 1, activation=activation)
        self.conv3 = tf.keras.layers.Conv2D(c1, 1, activation=activation)
        self.conv4 = tf.keras.layers.Conv2D(c1, 1, activation=activation)
        self.conv5 = tf.keras.layers.Conv2D(c1, 1, activation=activation)
        self.conv6 = tf.keras.layers.Conv2D(c1, 1, activation=activation)
        self.conv7 = tf.keras.layers.Conv2D(c2, 1, activation=activation)
        self.conv8 = tf.keras.layers.Conv2D(c2, 1, activation=activation)
        #self.conv9 = tf.keras.layers.Conv2D(c2, 1, activation=activation)
        #self.conv10 = tf.keras.layers.Conv2D(c2, 1, activation=activation)
        #self.conv11 = tf.keras.layers.Conv2D(c2, 1, activation=activation)
        #self.conv12 = tf.keras.layers.Conv2D(c2, 1, activation=activation)
        print(f"{self.__class__.__name__}:")
        print("fields:", self.fields)
        print("embed channels:", emb_ch)
        print("bins:", self.stat_channels // 3)
        print("c1, c2:", c1, c2)

    def stat(self, img):
        # Implement a statistics of img that outputs a tensor of shape [N, h, w, stat_channels]
        return None

    def call(self, x0, inputs):
        # devide input image into same receptive fields as x
        # x: [N, h, w, C]
        # img: [N, H, W, 3] -> stat: [N, h, w, stat_channels]
        stat = self.stat(img=inputs[0])

        #x = tf.concat([x0, stat], axis=-1, name="statpool_merge") if x0 is not None else stat  # merge-4-skipmerge-4
        x = stat  # all others

        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.conv3(x)
        x = self.bn4(x)

        # 6-skipmerge
        x = self.conv4(x)

        # 4-skipmerge-4
        #x = self.conv4(x)
        #x = tf.concat([x, x0, stat], axis=-1, name="statpool_skipmerge")  # merge-4-skipmerge-4
        #x = tf.concat([x, stat, x0], axis=-1, name="statpool_skipmerge")  # all others


        # 4-skip-4-(skip)merge
        #x4 = self.conv4(x)
        #x = tf.concat([x4, stat], axis=-1, name="statpool_skip")

        x = self.bn5(x)
        x = self.conv5(x)
        x = self.bn6(x)
        x = self.conv6(x)

        # 6-skipmerge
        x = tf.concat([x, x0, stat], axis=-1, name="statpool_skipmerge") if x0 is not None else x

        x = self.bn7(x)
        x = self.conv7(x)
        x = self.bn8(x)
        x = self.conv8(x)

        # 4-skip-4-(skip)merge
        #x = tf.concat([x, x0], axis=-1, name="statpool_merge") if x0 is not None else x  # 4-skip-4-merge
        #x = tf.concat([x, x4, x0], axis=-1, name="statpool_merge") if x0 is not None else x
        #x = tf.concat([x, x4, stat, x0], axis=-1, name="statpool_merge") if x0 is not None else x
        #x = self.bn9(x)
        #x = self.conv9(x)
        #x = self.bn10(x)
        #x = self.conv10(x)
        #x = self.bn11(x)
        #x = self.conv11(x)
        #x = self.bn12(x)
        #x = self.conv12(x)
        return self.pool(x)


class QuantilePooling(StatPooling):
    def stat(self, img):
        h, w, = self.fields
        c = self.stat_channels
        _, W, H, C = img.shape
        agg_h, agg_w = H // h, W // w
        agg_numel = agg_h * agg_w
        img = tf.reshape(tf.transpose(tf.reshape(img, [-1, h, H // h, w, W // w, C]), [0, 1, 3, 5, 2, 4]), [-1, h, w, C, agg_numel])
        stats = tfp.stats.quantiles(img, num_quantiles=c // C - 1, axis=-1, interpolation='linear')
        return tf.reshape(tf.transpose(stats, [1, 2, 3, 4, 0]), (-1, h, w, c))


class HistogramPooling(StatPooling):
    "Dysfunct: tfp.stats.histogram does not compile with XLA (issue #1758)"
    def stat(self, img):
        h, w, = 12, 12 #self.fields
        c = 32 * 3 #self.stat_channels
        bins = 32 #self.stat_channels // 3
        W, H, C = 384, 384, 3#_, W, H, C = img.shape
        agg_h, agg_w = H // h, W // w
        agg_numel = agg_h * agg_w
        edges = tf.linspace(-0.5, 255.5, bins + 1) / 255  # first will be replaced by -inf
        #img = tf.reshape(tf.transpose(tf.reshape(img, [-1, h, H // h, w, W // w, C]), [0, 1, 3, 5, 2, 4]), [-1, h, w, C, agg_numel])
        img = tf.reshape(tf.transpose(tf.reshape(img, [64, h, H // h, w, W // w, C]), [0, 1, 3, 5, 2, 4]), [64, h, w, C, agg_numel])
        stats = tfp.stats.histogram(img, edges=edges, axis=4, extend_lower_interval=True, extend_upper_interval=True)
        return tf.reshape(tf.transpose(stats, [1, 2, 3, 4, 0]), (64, h, w, c))


# TF data pipeline --------------------------------------------------------

def get_curves(target, eps=0.1, alpha_scale=1, beta_decay=10):
    # target: (2, C) for gamma + bp, (3, C) for a, b, bp (beta-curve)
    # curves: (256, C)
    if target.shape[0] == 2:
        gamma, bp = target[0, :], target[1, :]  # iter over symbolic tensor not allowed!
        curves = tf.range(0., 256., delta=1., dtype=tf.float32)[:, None]
        curves = tf.math.pow(curves, gamma[None, :])
        curves = curves * ((255. - bp[None, :]) / 255.) + bp[None, :]
        return curves

    a, b, bp = target[0, :], target[1, :], target[2, :]
    alpha = 1 + alpha_scale * tf.exp(a)
    beta = 1 / (1 + tf.exp(-(beta_decay * b)))  # sigmoid
    blackpoint = 500 * bp
    y = tf.range(0., 256., delta=0.1, dtype=tf.float32)
    support = tf.clip_by_value(y, 1e-3, 255) / 255 * (1 - eps)
    support = support[:, None]
    alpha = alpha[None, :]
    beta = beta[None, :]
    blackpoint = blackpoint[None, :]
    xs = tf.math.pow(support, alpha - 1) * tf.math.pow(1 - support, beta - 1)  # pdf(support)
    xs = blackpoint + (255 - blackpoint) * xs / xs[-1, :]
    q = tf.range(0., 256., delta=1.0, dtype=tf.float32)
    curves = [interp1d_tf(xs[:, c], y, q) for c in range(3)]
    return tf.stack(curves, axis=-1)

def curve_tfm(image, target):
    # image: channel last
    # target: (2, C) for gamma + bp, (3, C) for a, b, bp (beta-curve)
    gamma, bp = target[0, :], target[1, :]  # iter over symbolic tensor not allowed!
    bp = bp[None, None, :] / 255.
    image = tf.math.pow(image, gamma[None, None, :])
    image = image * (1. - bp) + bp
    return image

def curve_tfm6(image, target):
    # image: channel last
    # target: (2, C) for gamma + bp, (3, C) for a, b, bp (beta-curve)
    gamma, bp = target[0, :], target[1, :]
    bp = bp[None, :] / 255.
    x = tf.range(256, dtype=tf.float32)
    ys = tf.math.pow(x[:, None], gamma[None, :]) * (1. - bp) + bp
    #channels = [interp1d_tf(x, ys[:, c], image[:, :, c]) for c in range(3)]  # super slow
    #return tf.stack(channels, axis=-1)
    return tf.math.pow(image, gamma[None, None, :]) * (1. - bp[None, :, :]) + bp[None, :, :]

def interp1d_tf(x, y, inp):    
    # Find the interpolation indices
    c = tf.math.count_nonzero(tf.expand_dims(inp, axis=-1) >= x, axis=-1)
    idx0 = tf.maximum(c - 1, 0)
    idx1 = tf.minimum(c, tf.size(x, out_type=c.dtype) - 1)
    
    # Get interpolation X and Y values
    x0 = tf.gather(x, idx0)
    x1 = tf.gather(x, idx1)
    f0 = tf.gather(y, idx0)
    f1 = tf.gather(y, idx1)
    
    # Compute interpolation coefficient
    x_diff = x1 - x0
    alpha = (inp - x0) / tf.where(x_diff > 0, x_diff, tf.ones_like(x_diff))
    alpha = tf.clip_by_value(alpha, 0, 1)
    
    # Compute interpolation
    return f0 * (1 - alpha) + f1 * alpha


def map_index(image, curves, height=None, width=None, channels=3, 
              add_uniform_noise=False, add_jpeg_artifacts=True, sharpness_augment=True,
              resize_before_jpeg=False):
    """Transform `image` by mapping its pixel values via `curves`
    
    Parameters:
        image (tf.Tensor) Input image with uint8 pixel values, shape (H, W, C)
        curves (tf.Tensor): Transformation curves for each channel of the image
        add_uniform_noise (bool, optional): Erase uint8 quantization steps with uniform noise,
        height (int): Height of the input image. Required if add_uniform_noise.
        width (int): Width of the input image. Required if add_uniform_noise.
    
    Returns:
        tf.Tensor: Output image with mapped pixel values (float32)
    """
    # OperatorNotAllowedInGraphError: Iterating over a symbolic `tf.Tensor` is not allowed
    #for i, curve in enumerate(curves):
    #    image[:, :, i] = curve[image[:, :, i]]
    # No fancy indexing in TF:
    #for i in range(3):
    #    image[:, :, i] = curves[:, i][image[:, :, i]]
    # Use tf.gather_nd instead:

    if add_jpeg_artifacts:
        # is this faster than uint8 conversion after mapping by tf.image.adjust_jpeg_quality?
        curves = tf.cast(tf.clip_by_value(curves, 0, 1) * 255, tf.uint8)
        assert not add_uniform_noise, 'add_jpeg_artifacts and add_uniform_noise are mutually exclusive!'
    else:
        curves = tf.cast(curves, tf.float32)
    #curves = tf.cast(curves, tf.float32)

    # turn pixel values into int32 indices for gather_nd
    image = tf.cast(image, tf.int32)
    channel_idx = tf.ones_like(image, dtype=tf.int32) * tf.range(3)[None, None, :]
    indices = tf.stack([image, channel_idx], axis=-1)
    if add_uniform_noise:
        image_plus_one = tf.clip_by_value(image + 1, 0, 255)
        indices_plus_one = tf.stack([image_plus_one, channel_idx], axis=-1)

    image = tf.gather_nd(curves, indices)

    if resize_before_jpeg:
        # resize only images larger than cfg.size
        max_height, max_width = resize_before_jpeg
        height = tf.clip_by_value(height, 16, max_height)
        width = tf.clip_by_value(width, 16, max_width)

        # resize all images (worse)
        #height, width = resize_before_jpeg

        # resize converts uint8 -> float32
        image = tf.cast(tf.image.resize(image, (height, width)), tf.uint8)

    if sharpness_augment:
        # randomly soften/sharpen the image
        #rnd_sharpness = tf.exp(-2.3 + 4.6 * tf.random.uniform([]))
        rnd_sharpness = 2.0 * tf.random.uniform([])  # rng checked
        image = adjust_sharpness_tf(image, rnd_sharpness)

    if add_jpeg_artifacts:
        # adjust_jpeg_quality automatically converts image to uint8 and back
        rnd_quality = tf.cast(50 * (1 + tf.random.uniform([])), tf.int32)
        image = tf.image.adjust_jpeg_quality(image, rnd_quality)
        image = tf.cast(image, tf.float32) / 255

    if add_uniform_noise:
        image_plus_one = tf.gather_nd(curves, indices_plus_one)
        noise_range = image_plus_one - image  ## uint - float
        noise = tf.random.uniform((height, width, channels)) * noise_range
        image += noise

    return image, height, width


def preprocess_image(image, preprocess):
    # apply dataset-specific normalization
    image = tf.cast(image, tf.float32) / 255.0
    if 'bp' in preprocess:
        bp = preprocess['bp']
        image = bp + (1 - bp) * image
    if 'gamma' in preprocess:
        gamma = preprocess['gamma']
        image = tf.pow(tf.clip_by_value(image, 1e-6, 1), gamma)
    if ('a' in preprocess) and ('b' in preprocess):
        a, b = preprocess['a'], preprocess['b']
        image = 1 - tf.cos(pi_half * tf.clip_by_value(image, 1e-6, 1 - 1e-6) ** a) ** b
    if 'bp2' in preprocess:
        bp2 = preprocess['bp2']
        image = bp2 + (1 - bp2) * image

    # restore bias from reference dataset (inverse transform) 
    if 'bp2_ref' in preprocess:
        bp = preprocess['bp2'] / (preprocess['bp2'] - 1)
        image = bp + (1 - bp) * image
    if 'gamma_ref' in preprocess:
        gamma = 1 / preprocess['gamma_ref']
        image = tf.pow(tf.clip_by_value(image, 1e-6, 1), gamma)
    if 'bp_ref' in preprocess:
        bp2 = preprocess['bp_ref'] / (preprocess['bp_ref'] - 1)
        image = bp2 + (1 - bp2) * image

    return tf.cast(tf.clip_by_value(image * 255, 0, 255), tf.uint8)


def curve_tfm_image(cfg, image, tfm, height, width):
    if cfg.curve == 'beta':
        curves = get_curves(tfm, alpha_scale=cfg.alpha_scale or 1, beta_decay=cfg.beta_decay or 10)  # (256, C)

        if height is None or width is None:
            height, width = [int(s * cfg.presize) for s in cfg.size]
            image = tf.image.resize(image, [height, width])
        image, height, width = map_index(image, curves, cfg.add_uniform_noise, height, width)

    elif cfg.curve in ['gamma', 'free']:
        curves = tf.transpose(tfm)  # (256, C)

        if height is None or width is None:
            assert cfg.presize, f'neither image height/width nor cfg.presize is defined'
            height, width = [int(s * cfg.presize) for s in cfg.size]
            image = tf.image.resize(image, [height, width])

        assert not (cfg.resize_before_jpeg and cfg.catmix), 'catmix and resize_before_jpeg are mutually exclusive'

        image, height, width = map_index(image, curves, height, width, 3,
                                         cfg.add_uniform_noise, cfg.add_jpeg_artifacts, cfg.sharpness_augment,
                                         resize_before_jpeg=cfg.size if cfg.resize_before_jpeg else False)

    if cfg.curve not in ['gamma', 'free']:
        image /= 255.0

    if cfg.noise_level:
        rnd_factor = tf.random.uniform(())
        image += cfg.noise_level * rnd_factor * tf.random.normal((height, width, 3))
    
    return tf.clip_by_value(image, 0, 1)  # does it matter?


def decode_image(cfg, image_data, tfm, height, width):
    "Decode image and apply curve transform tfm"
    # Cannot use image.shape: ValueError: Cannot convert a partially known TensorShape (None, None, 3) to a Tensor
    # => read height, width features from tfrec.
    
    image = tf.image.decode_jpeg(image_data, channels=3)
    
    if cfg.BGR:
        image = tf.reverse(image, axis=[-1])

    if cfg.normalize in ['torch', 'tf', 'caffe']:
        from keras.applications.imagenet_utils import preprocess_input
        return preprocess_input(image, mode=cfg.normalize)
    elif isinstance(cfg.preprocess, dict):
        image = preprocess_image(image, cfg.preprocess)

    if isinstance(cfg.curve, str) and not cfg.catmix:
        return curve_tfm_image(cfg, image, tfm, height, width)

    return tf.cast(image, tf.float32) / 255.0


def get_mask_uniform(ps, n_channels=3):
    """Returns float32 mask with shape [len(ps), n_channels]
    
    Elements are either 0 or 1 and sum up to 1 over axis 0."""
    ps = (np.array(ps) / np.sum(ps)).tolist()
    edges = tf.constant(np.cumsum([0] + ps, dtype=np.float32))
    r = tf.random.uniform([n_channels])
    mask = tf.stack([tf.less_equal(edges[i], r) & tf.less(r, edges[i+1]) for i in range(3)])
    return tf.cast(mask, tf.float32)

 
def get_mask_categorical(ps, n_channels=3):
    """Returns float32 mask with shape [len(ps), n_channels]
    
    Elements are either 0 or 1 and sum up to 1 over axis 0."""
    samples = tf.random.categorical(tf.math.log([ps]), n_channels)[0]
    mask = tf.one_hot(samples, depth=len(ps), axis=0)
    return mask


def get_mask_mix(ps, n_channels=3, n_samples=5):
    """Returns float32 mask with shape [len(ps), n_channels]
    
    Elements are in range 0...1 and sum up to 1 over axis 0.
    n_samples: number of draws to be averaged for each channel
    Deprecated, yields kinky unnatural curves."""
    samples = tf.random.categorical(tf.math.log([ps]), n_channels * n_samples)[0]
    mask = tf.one_hot(samples, depth=len(ps), axis=0)
    mask = tf.reduce_mean(tf.reshape(mask, (-1, n_channels, n_samples)), axis=-1)
    return mask


def beta_pdf(x, alpha, beta):
    x = tf.clip_by_value(x, 1e-6, 1 - 1e-6)  # avoid nan
    x = x * 0.9                              # reduce slope at whitepoint
    x = tf.pow(x, alpha - 1) * tf.pow(1 - x, beta - 1)  # unnormalized PDF(x)
    x /= x[:, -1:]                                      # normalize
    return x


def inverse_beta_pdf(x, alpha, beta):
    # float64 avoids div-by-zero below
    y = tf.linspace(1e-6, 1 - 1e-6, 2000)
    y = tf.cast(y, tf.float64)
    x = tf.clip_by_value(x, 0, 1)
    x = tf.cast(x, tf.float64)
    alpha = tf.cast(alpha, tf.float64)
    beta = tf.cast(beta, tf.float64)
    pdfs = beta_pdf(y[None, :], alpha, beta)
    #pdfs = tf.clip_by_value(pdfs, 0, 1)  # probably not necessary

    assert x.shape == [3, 256], f'unexpected x.shape {x.shape}'
    # this was assuming x.shape = [256]:
    #x = tf.stack([interp1d_tf(pdfs[i], y, x) for i in range(3)])

    # if we get x with channel components x.shape=[3, 256]:
    x = tf.stack([interp1d_tf(pdfs[i], y, x[i]) for i in range(3)])

    return tf.cast(tf.clip_by_value(x, 0, 1), tf.float32)


def check_reversed_tfm(tfm, target):
    # Does not work with symbolic tensors (inside the data pipeline)
    image = tf.transpose(tfm)[None, :, :]  # [1, 256, 3]
    reversed_tfm, _, _ = map_index(image, target, 1, 256, 3, False, False, False, False)  # HWC
    assert reversed_tfm.shape == [1, 256, 3], f'{reversed_tfm.shape}'
    reversed_tfm = tf.transpose(reversed_tfm[0, :, :])  # [C, 256]
    rmse = tf.math.reduce_std(reversed_tfm - tf.linspace(0.0, 1.0, 256)[None, :])
    if (rmse > 1) and hasattr(tfm, 'numpy'):
        fn = 'tfm_target.csv'
        tf.print(f"reversed_tfm has rmse of {rmse}, saving {fn}")
        #with open(f'../{fn}', 'w') as fp:
        #    for i in range(256):
        #        s = ','.join(f'{tfm[c, i].numpy():.4f},{target[c, i].numpy():.4f}' for c in range(3))
        #        fp.write(s + '\n')
        #raise ValueError("stopping for debug")


def curve4(x, a, b):
    x = tf.clip_by_value(x, 1e-6, 1 - 1e-6)  # avoid nan
    return 1 - tf.cos(pi_half * x ** a) ** b


def inverse_curve4(x, a, b):
    x = tf.clip_by_value(x, 1e-6, 1 - 1e-6)  # avoid nan
    return pi_half**(-1 / a) * tf.math.acos((1 - x)**(1 / b))**(1 / a)


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

    # tf.random.uniform is 8x faster than tfd.Uniform.sample
    if cfg.curve == 'gamma':
        target_bp = tf.random.uniform([3], *cfg.blackpoint_range) / 255
        target_bp2 = tf.random.uniform([3], *cfg.blackpoint2_range) / 255
        target_log_gamma = tf.random.uniform([3], *cfg.log_gamma_range)

        # calculate target_curve, tfm
        support = tf.linspace(0.0, 1.0, 256)
        bp = target_bp[:, None]
        bp2 = target_bp2[:, None]
        log_gamma = target_log_gamma[:, None]
        gamma = tf.exp(log_gamma)
        x = support[None, :]
        x = (x - bp2) / (1 - bp2)
        x = tf.clip_by_value(x, 1e-6, 1)  # avoid nan
        x = tf.pow(x, 1 / gamma)
        x = (x - bp) / (1 - bp)
        tfm = tf.clip_by_value(x, 0, 1)
        if cfg.predict_inverse:
            x = support[None, :]
            x = bp + x * (1 - bp)
            x = tf.clip_by_value(x, 1e-6, 1)  # avoid nan
            x = tf.pow(x, gamma)
            x = x * (1 - bp2) + bp2
            target_curve = tf.clip_by_value(x, 0, 1)
        else:
            target_curve = tfm

    elif cfg.curve == 'beta' and cfg.channel_size == 2:
        # target: (2, C) for gamma + bp, (3, C) for a, b, bp (beta-curve)
        gamma = tf.math.exp(tfd.Normal(0.0, 0.4).sample([3]))
        bp = tfd.HalfNormal(scale=40.).sample([3])
        target = tf.stack([gamma, bp])
    elif cfg.curve == 'beta':
        # do not instantiate tfd classes globally: TF allocates half of GPU!
        dist_a = tfd.Normal(0.0, scale=cfg.a_sigma or 0.5)
        dist_b = tfd.Normal(cfg.b_mean or 0.4, scale=cfg.b_sigma or 0.25)
        dist_bp = tfd.HalfNormal(scale=cfg.bp_sigma or 0.02)
        a = dist_a.sample([3])
        b = dist_b.sample([3])
        bp = dist_bp.sample([3])
        target = tf.stack([a, b, bp])
    elif cfg.curve == 'free':
        # Cannot use numpy classes:
        #   - np.random produces identical params for all examples!
        #   - TF.random cannot seed numpy because no eager mode inside graph, tensors have no numpy()
        #
        # ToDo: factor out curve generation
        #
        # Issue: curve types not mixed during training, near-linear channels under-represented
        # Option1: randomly replace Beta/Curve4 by (small) gamma for individual channels
        # Option2: randomly apply one curve-type per channel
        # Option3: random linear combination of each curve type
        # Option4: apply Beta/Curve4 tfm after gamma tfm on 0-3 channels

        support = tf.linspace(0.0, 1.0, 256)
        bp = tf.random.uniform([3], *cfg.blackpoint_range)[:, None] / 255
        bp2 = tf.random.uniform([3], *cfg.blackpoint2_range)[:, None] / 255
        bp_clip = max(*cfg.blackpoint_range, *cfg.blackpoint2_range) if cfg.clip_target_blackpoint else None
        if bp_clip:
            bp_clip = max(int(bp_clip), 0)

        # Curve0 component: gamma
        log_gamma = tf.random.uniform([3], *cfg.log_gamma_range)
        gamma = tf.exp(log_gamma)[:, None]

        x = support[None, :]
        x = (x - bp2) / (1 - bp2)
        x = tf.clip_by_value(x, 1e-6, 1)  # avoid nan
        x = tf.pow(x, 1 / gamma)

        if bp_clip:
            bp_max = x[:, bp_clip:bp_clip + 1]
            _bp = tf.math.minimum(bp, bp_max)
        else:
            _bp = bp

        x = (x - _bp) / (1 - _bp)
        tfm0 = tf.clip_by_value(x, 0, 1)

        if cfg.predict_inverse:
            x = support[None, :]
            x = _bp + x * (1 - _bp)
            x = tf.clip_by_value(x, 1e-6, 1)  # avoid nan
            x = tf.pow(x, gamma)
            x = x * (1 - bp2) + bp2
            target0 = tf.clip_by_value(x, 0, 1)
        else:
            target0 = tfm0

        # Curve3 component: Beta-distribution PDF
        a = tf.random.uniform([3], *cfg.curve3_a_range)
        alpha = tf.exp(a)[:, None]
        beta = tf.random.uniform([3], *cfg.curve3_beta_range)[:, None]
        if cfg.predict_inverse and cfg.mirror_beta:
            # randomly and channel-wise mirror beta PDF (keep bp/bp2)
            mask = tf.cast(tf.random.uniform([3, 1], minval=0, maxval=2, dtype=tf.int32), tf.float32)

        # whether clipped or not, _bp2 is needed in any case
        if cfg.predict_inverse or cfg.bp_clip:
            x = support[None, :]
            x = bp + x * (1 - bp)
            if cfg.predict_inverse and cfg.mirror_beta:
                x = beta_pdf(x, alpha, beta) * mask + inverse_beta_pdf(x, alpha, beta) * (1 - mask)
            else:
                x = beta_pdf(x, alpha, beta)
            if bp_clip:
                bp2_max = (bp_clip / 255 - x[:, 0:1]) / (1 - x[:, 0:1])
                _bp2 = tf.math.minimum(bp2, bp2_max)
            else:
                _bp2 = bp2
            x = x * (1 - _bp2) + _bp2
            target1 = tf.clip_by_value(x, 0, 1)
        else:
            _bp2 = bp2

        x = support[None, :]
        x = (x - _bp2) / (1 - _bp2)
        if cfg.predict_inverse and cfg.mirror_beta:
            x = inverse_beta_pdf(x, alpha, beta) * mask + beta_pdf(x, alpha, beta) * (1 - mask)
        else:
            x = inverse_beta_pdf(x, alpha, beta)
        x = (x - bp) / (1 - bp)
        tfm1 = tf.clip_by_value(x, 0, 1)

        if not cfg.predict_inverse:
            target1 = tfm1

        # Curve4 component
        a = tf.exp(tf.random.uniform([3], *cfg.curve4_loga_range))[:, None]
        b = tf.random.uniform([3], *cfg.curve4_b_range)[:, None]
        if cfg.predict_inverse and cfg.mirror_curve4:
            # randomly and channel-wise mirror curve4 (keep bp/bp2)
            mask = tf.cast(tf.random.uniform([3, 1], minval=0, maxval=2, dtype=tf.int32), tf.float32)

        x = support[None, :]
        x = (x - bp2) / (1 - bp2)
        if cfg.predict_inverse and cfg.mirror_curve4:
            x = inverse_curve4(x, a, b) * mask + curve4(x, a, b) * (1 - mask)
        else:
            x = inverse_curve4(x, a, b)
        if bp_clip:
            bp_max = x[:, bp_clip:bp_clip + 1]
            _bp = tf.math.minimum(bp, bp_max)
        else:
            _bp = bp
        x = (x - _bp) / (1 - _bp)
        tfm2 = tf.clip_by_value(x, 0, 1)

        if cfg.predict_inverse:
            x = support[None, :]
            x = _bp + x * (1 - _bp)
            if cfg.predict_inverse and cfg.mirror_curve4:
                x = curve4(x, a, b) * mask + inverse_curve4(x, a, b) * (1 - mask)
            else:
                x = curve4(x, a, b)
            x = x * (1 - bp2) + bp2
            target2 = tf.clip_by_value(x, 0, 1)
        else:
            target2 = tfm2

        # Compose target, tfm from gamma, beta, curve4
        p_gamma = cfg.p_gamma
        p_beta = (1 - cfg.p_gamma) * cfg.p_beta
        p_curve4 = (1 - p_gamma - p_beta)
        #mask = get_mask_uniform([p_gamma, p_beta, p_curve4])
        mask = get_mask_categorical([p_gamma, p_beta, p_curve4])  # faster on cpu
        tfms = tf.stack([tfm0, tfm1, tfm2], axis=-1)
        targets = tf.stack([target0, target1, target2], axis=-1)
        target = tf.einsum('ijk,ki->ij', targets, mask)
        tfm = tf.einsum('ijk,ki->ij', tfms, mask)

        target = tf.reshape(target, [cfg.channel_size])
        #assert target.shape == (3 * 256,), f"wrong target shape: {target.shape}"
        #assert target.dtype == tf.float32, f"wrong target dtype: {target.dtype}"

        #tf.print("tfm:", tfm.shape, tf.reduce_min(tfm), tf.reduce_mean(tfm), tf.reduce_max(tfm))  # tfm: TensorShape([3, 256]) 0 0.598787069 1



    if 'height' in cfg.data_format:
        height = example[cfg.data_format['height']]
        width = example[cfg.data_format['width']]
    else:
        height, width = (
            (512, 512) if '-512-' in cfg.datasets[0] else 
            (512, 512) if (cfg.datasets[0] == 'flickrfacestfrecords') else
            (256, 256) if 'places365' in cfg.tags else
            (None, None))

    features['image'] = decode_image(cfg, example[cfg.data_format['image']], tfm,
                                     height, width)

    if cfg.curve == 'beta':
        # split target into 2 (3) components for weighted loss
        if cfg.channel_size == 6:
            features['target_gamma'] = target[0, :]
            features['target_bp'] = target[1, :]
        else:
            features['target_a'] = target[0, :]
            features['target_b'] = target[1, :]
            features['target_bp'] = target[2, :]

    elif cfg.curve == 'free':
        features['target'] = target

    elif cfg.curve == 'gamma':
        features['target_curve'] = target_curve
        features['target_bp'] = target_bp
        features['target_bp2'] = target_bp2
        features['target_log_gamma'] = target_log_gamma

    if cfg.catmix and isinstance(cfg.curve, str):
        # apply tfm in post_catmix
        features['tfm'] = tfm
        features['height'] = height
        features['width'] = width

    # tf.keras.model.fit() wants dataset to yield a tuple (inputs, targets, [sample_weights])
    # inputs can be a dict
    inputs = tuple(features[key] for key in cfg.inputs)
    targets = tuple(features[key] for key in cfg.targets)

    return inputs, targets


def post_catmix(cfg, inputs, targets):
    # apply curve tfm
    target, tfm = targets
    height, width = cfg.size  # fixed size of images composed by catmix
    image = inputs[0]
    image = tf.cast(tf.clip_by_value(image * 255, 0, 255), tf.uint8)
    image = curve_tfm_image(cfg, image, tfm, height, width)
    return (image,), (target,)


def count_examples_in_tfrec(fn):
    count = 0
    for _ in tf.data.TFRecordDataset(fn):
        count += 1
    return count


def count_data_items(filenames, tfrec_filename_pattern=None):
    if '-of-' in filenames[0]:
        # Imagenet: Got number of items from idx files
        #return sum(391 if 'validation' in fn else 1252 for fn in filenames)

        # imagenet-selection-*: same filenames as Imagenet but varying items per file
        return 365269 if (len(filenames) == 921) else 88658  # for n_folds=5
    if 'FlickrFaces' in filenames[0]:
        # flickr: number of items in filename
        return sum(int(fn[-10:-6]) for fn in filenames)
    if 'coco' in filenames[0]:
        return sum(159 if 'train/coco184' in fn else 499 if 'val/coco7' in fn else 
                   642 if '/train/' in fn else 643 
                   for fn in filenames)
    if 'landmark-2021-gld2' in filenames[0]:
        return sum(
            203368 if 'gld2-0' in fn else 203303 if 'gld2-1' in fn else 
            203376 if 'gld2-2' in fn else 203316 if 'gld2-4' in fn else 
            203415 if 'gld2-7' in fn else 203400 if 'gld2-8' in fn else
            203321 if 'gld2-9' in fn else 0  # gld2-3 is corrupt, 5 and 6 missing
            for fn in filenames)
    if all('train_' in fn for fn in filenames) and len(filenames) in [45, 6]:
        # landmark2021: no subfolders or identifiable string in urls
        # if too large, valid raises out-of-range error
        #return 1448943 if len(filenames) == 45 else 193665  # landmark2021
        return 574999 if len(filenames) == 45 else 76838  # landmark2021sele
    if all('train_' in fn for fn in filenames) and len(filenames) in [46, 4]:
        # landmark2020: no subfolders or identifiable string in urls
        # if too large, valid raises out-of-range error
        # all 31610 except train_20 ...31 (31609)
        return 31610 * 46 if len(filenames) == 46 else 31609 * 4
    if Path(filenames[0]).stem[3] == '-':
        # places365-tfrec: number of items in filename
        return sum(int(Path(fn).stem[4:]) for fn in filenames)
    if tfrec_filename_pattern is not None:
        "Infer number of items from tfrecord file names."
        tfrec_filename_pattern = tfrec_filename_pattern or r"-([0-9]*)\."
        pattern = re.compile(tfrec_filename_pattern)
        n = [int(pattern.search(fn).group(1)) for fn in filenames if pattern.search(fn)]
        if len(n) < len(filenames):
            print(f"WARNING: only {len(n)} / {len(filenames)} urls follow the convention:")
            for fn in filenames:
                print(fn)
        return sum(n)
    else:
        if True:
            # count them (slow)
            print(f'Counting examples in {len(filenames)} urls:')
            total_count = 0
            for i, fn in enumerate(filenames):
                n = count_examples_in_tfrec(fn)
                #print(f"   {i:3d}", Path(fn).parent, n)
                print(f"   {i:3d}", fn, n)
                total_count += n
            return total_count
        else:
            raise NotImplementedError(f'autolevels.count_data_items: filename not recognized: {filenames[0]}')


@tf.keras.utils.register_keras_serializable()
class TFCurveRMSE(tf.keras.metrics.MeanSquaredError):
    def __init__(self, name='curve_rmse', curve='gamma', **kwargs):
        if curve == 'gamma': name = 'rmse'
        super().__init__(name=name, **kwargs)
        self.curve = curve

    def get_config(self):
        return dict(name=self.name, curve=self.curve, dtype=self.dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if False and (self.curve == 'gamma'):
            # (256,) ** (N, 3) -> (N, 3, 256)
            support = tf.cast(tf.linspace(0, 1, 256), dtype=self._dtype)
            y_true = tf.pow(support[None, None, :], tf.exp(y_true)[:, :, None])
            y_pred = tf.pow(support[None, None, :], tf.exp(y_pred)[:, :, None])

        return super().update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        return tf.sqrt(super().result()) * 255


class CurveRMSE(MeanSquaredError):
    def __init__(self, curve='gamma', squared=False):
        super().__init__(squared=squared)
        self.curve = curve

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        support = torch.linspace(0, 1, 256, dtype=preds.dtype)
        if self.curve == 'gamma':
            # (256,) ** (N, 3) -> (N, 3, 256)
            target = torch.pow(support[None, None, :], torch.exp(target)[:, :, None])
            preds = torch.pow(support[None, None, :], torch.exp(preds)[:, :, None])

        return super().update(preds, target)
    
    def compute(self):
        return super().compute() * 255


def on_train_end(cfg, model, metrics):
    if ('tf' in cfg.tags) and isinstance(cfg.preprocess, type) and ('preprocess' not in cfg.freeze):
        # print trained preprocess parameters
        print(f"\nPreprocess weights:")
        for w in model.layers[1].weights:
            if w.shape != (3,): continue
            print(f"    {w.name:<20} {', '.join([f'{x:9.6f}' for x in w.numpy()])}")
