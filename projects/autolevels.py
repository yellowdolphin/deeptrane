import os
from pathlib import Path
from functools import partial
import numpy as np
import pandas as pd
import cv2
print("import torch...")
import torch
from torch.utils.data import Dataset
import torchvision
print("[ √ ] torchvision:", torchvision.__version__)
import torchvision.transforms as TT
from torchvision.transforms.functional import InterpolationMode, resize
from torchvision.io import encode_jpeg, decode_jpeg
from scipy.interpolate import interp1d

try:
    import torchmetrics as tm
    from torchmetrics import MeanSquaredError
except ModuleNotFoundError:
    from utils.general import quietly_run
    quietly_run('pip install torchmetrics')
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
from albumentations.augmentations.geometric.resize import LongestMaxSize


import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from augmentation import adjust_sharpness_tf


def init(cfg):
    cfg.competition_path = Path('/kaggle/input/imagenet-object-localization-challenge')
    if cfg.cloud == 'drive':
        cfg.competition_path = Path(f'/content/gdrive/MyDrive/{cfg.project}')

    if cfg.filetype == 'JPEG':
        cfg.image_root = (
            cfg.image_root if (cfg.cloud == 'drive') else
            cfg.competition_path / 'ILSVRC/Data/CLS-LOC/train')
    elif 'imagenet' in cfg.tags:
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

    cfg.meta_csv = (
        cfg.meta_csv if (cfg.cloud == 'drive') else
        cfg.competition_path / 'ILSVRC/ImageSets/CLS-LOC/train_cls.txt')

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
        cfg.dataset_class = AugInvCurveDataset3
        cfg.channel_size = 256 * 3
        cfg.targets = ['target']


def read_csv(cfg):
    if cfg.DEBUG:
        return pd.read_csv(cfg.meta_csv, sep=' ', usecols=[0], header=None, names=['image_id']).sample(frac=0.01)
    return pd.read_csv(cfg.meta_csv, sep=' ', usecols=[0], header=None, names=['image_id'])


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


def map_index_torch(image, tfm, add_uniform_noise=False):
    # map image (H, W, C) to tfm (C, 256) using torch.gather (faster than numpy fancy indexing)
    # tfm must be expanded to have same shape as image (except dim=1)
    expanded_curves = torch.tensor(tfm.T)[None, :, :].expand(image.shape[0], -1, -1)
    if not add_uniform_noise:
        return torch.gather(expanded_curves, dim=1, index=torch.LongTensor(image))
    
    # add uniform noise to mask uint8 quantization (slow)
    image_plus_one = torch.gather(expanded_curves, dim=1, index=(torch.LongTensor(image) + 1).clamp(max=255))    
    image = torch.gather(expanded_curves, dim=1, index=torch.LongTensor(image))
    return image + torch.rand_like(image) * (image_plus_one - image)


class Curve0():
    def __init__(self, gamma=1.0, bp=0, bp2=0, unclipped=False):
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
        self.unclipped = bool(unclipped)

    def __call__(self, x):
        assert (x.shape[0] in {1, 3}) and (x.ndim > 1), f'x has wrong shape: {x.shape}, expecting (C, *)'
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
        x = (x - bp) / (1 - bp)
        return x if self.unclipped else np.clip(x, 0, 1)


class Curve3():
    def __init__(self, alpha=2.0, beta=0.99, bp=0, bp2=0, unclipped=False):
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
        self.unclipped = bool(unclipped)

    def __call__(self, x):
        assert (x.shape[0] in {1, 3}) and (x.ndim > 1), f'x has wrong shape: {x.shape}, expecting (C, *)'
        alpha, beta, bp, bp2 = (p[:, None] for p in self.params)

        x = bp + x * (1 - bp)
        x = np.clip(x, self.x_min, self.x_max)  # avoid nan
        x = x * (1 - self.eps)                  # reduce slope at whitepoint
        x = np.power(x, alpha - 1) * np.power(1 - x, beta - 1)  # unnormalized PDF(x)
        x /= x[:, -1:]                                          # normalize
        x = x * (1 - bp2) + bp2
        return x if self.unclipped else np.clip(x, 0, 1)
    
    def inverse(self, xs):
        assert (xs.shape[0] in {1, 3}) and (xs.ndim > 1), f'x has wrong shape: {xs.shape}, expecting (C, *)'

        y = np.linspace(self.x_min, self.x_max, 2000, dtype=np.float64)  # float64 avoids div-by-zero below
        pdfs = self.__call__(y[None, :])
        assert pdfs.shape[0] == 3, str(pdfs.shape)
        xs = xs.repeat(3, axis=0) if xs.shape[0] == 1 else xs
        # fill_value='extrapolate' produces NaNs
        return np.stack([interp1d(pdf, y, fill_value=(x[0], x[-1]), bounds_error=False,
                                  assume_sorted=True)(x).clip(0, 1).astype(xs.dtype) for x, pdf in zip(xs, pdfs)])


π_half = 0.5 * np.pi
class Curve4():
    def __init__(self, a=0.5, b=0.81, bp=0, bp2=0, unclipped=False):
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
        self.unclipped = bool(unclipped)

    def __call__(self, x):
        assert (x.shape[0] in {1, 3}) and (x.ndim > 1), f'x has wrong shape: {x.shape}, expecting (C, *)'
        a, b, bp, bp2 = (p[:, None] for p in self.params)

        x = bp + x * (1 - bp)
        x = np.clip(x, self.x_min, self.x_max)  # avoid nan
        x = 1 - np.cos(π_half * x ** a) ** b
        x = x * (1 - bp2) + bp2
        return x if self.unclipped else np.clip(x, 0, 1)
    
    def inverse(self, x):
        assert (x.shape[0] in {1, 3}) and (x.ndim > 1), f'x has wrong shape: {x.shape}, expecting (C, *)'
        a, b, bp, bp2 = (p[:, None] for p in self.params)

        x = (x - bp2) / (1 - bp2)
        x = np.clip(x, self.x_min, self.x_max)  # avoid nan
        x = π_half**(-1 / a) * np.arccos((1 - x)**(1 / b))**(1 / a)
        x = (x - bp) / (1 - bp)
        return x if self.unclipped else np.clip(x, 0, 1)


class ObsoleteCurve():
    def __init__(self, a=1, b=1, bp=0, eps=0.1, alpha_scale=1, beta_decay=10, create_blackpoint=False):
        """Set of Beta distributions, PDF and inverse PDF

        a: log(alpha parameter)
        b: logit(beta parameter)
        bp: black point / 500
        eps: small positive (nonzero) value, increase to reduce peak slope
        create_blackpoint: mirror blackpoint to obtain more realistic photos"""
        super().__init__()
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float)
        self.bp = np.array(bp, dtype=float)
        assert self.a.shape == self.b.shape == self.bp.shape, 'parameter shape mismatch'
        self.eps = float(eps)
        self.alpha_scale = float(alpha_scale)
        self.beta_decay = float(beta_decay)
        self.max_x = 1 - self.eps
        self.create_blackpoint = create_blackpoint

    @staticmethod
    def match_dims(p, x):
        "Append new axes to p or prepend new axes to x until they match"
        # align first (batch/channel) axes (different parameter for each element)
        while not all([any([ap == ax, ap == 1, ax == 1]) for ap, ax in zip(p.shape, x.shape)]):
            x = x[None, ...]
        
        # append curve/height/width... axes to p (same parameters for all elements)
        dim_p = p.dim() if isinstance(p, torch.Tensor) else p.ndim
        dim_x = x.dim() if isinstance(x, torch.Tensor) else x.ndim
        while dim_p < dim_x:
            p, dim_p = p[..., None], dim_p + 1
        
        return p, x

    def prepare_broadcast(self, params, x):
        "Add extra dims to each of params to match those of x"
        params = list(params)
        for i, p in enumerate(params):
            params[i], x = self.match_dims(p, x)
        return params, x

    def pdf(self, x):
        "Input array value range: 0 ... 255"
        params = self.transformed_parameters()
        (alpha, beta, blackpoint), x = self.prepare_broadcast(params, x)
        x = np.clip(x, 1e-3, None) / 255 * (1 - self.eps)
        y = np.power(x, alpha - 1) * np.power(1 - x, beta - 1)
        # normalization depends on alpha, beta and must be calculated image/channel-wise
        hw_axes = tuple(i for i, (a, b) in enumerate(zip(alpha.shape, x.shape)) if b > a)
        y_norm = y.max(axis=hw_axes, keepdims=True)
        return blackpoint + (255 - blackpoint) * y / y_norm

    def transformed_parameters(self):
        alpha = 1 + self.alpha_scale * np.exp(self.a)
        beta = 1 / (1 + np.exp(-(self.beta_decay * self.b)))  # sigmoid
        blackpoint = 500 * self.bp
        return alpha, beta, blackpoint

    @classmethod
    def from_transformed_parameters(cls, alpha, beta, blackpoint, eps=0.1):
        a = np.log(alpha - 1)
        b = np.log(beta / (1 - beta)) / 10  # logit
        bp = blackpoint / 500
        return cls(a, b, bp, eps)

    def get_inverse_curves(self, channel_last=False):
        y = np.linspace(0, 255, 10000)
        xs = self.pdf(y)
        curves = []
        for x in xs:
            if self.create_blackpoint:
                # resembles more realistic photo
                f = interp1d(x, y, fill_value=(x[0], x[-1]), bounds_error=False, assume_sorted=True)
                support = np.arange(256)
            else:
                f = interp1d(x, y, bounds_error=True, assume_sorted=True)
                support = np.arange(256).clip(x[0], x[-1])
            curves.append(f(support))
        
        return np.stack(curves, axis=-1) if channel_last else np.stack(curves)
    
    def apply_inverse_pdf(self, img):
        y = np.linspace(0, 255, 10000)
        xs = self.pdf(y)
        for i, x in enumerate(xs):
            if self.create_blackpoint:
                # resembles more realistic photo
                f = interp1d(x, y, fill_value=(x[0], x[-1]), bounds_error=False, assume_sorted=True)
                img[:, :, i] = f(img[:, :, i]).astype(img.dtype)
            else:
                f = interp1d(x, y, bounds_error=True, assume_sorted=True)
                img[:, :, i] = f(img[:, :, i].clip(x[0], x[-1])).astype(img.dtype)
        return img            


π_half = 0.5 * np.pi
class ObsoleteCurve4():
    def __init__(self, a=0.5, b=0.81, bp=0, bp2=0, inverse=False):
        """Function y(x) = 1 - cos(π/2 * x^a)^b  with offsets bp, bp2 in x, y

        If `inverse`, returns the inverse function 
            y^-1(x) = (2 / π)**(1 / a) * np.arccos((1 - x)**(1 / b))**(1 / a)

        Input/Output range: [0, 1]

        Parameters
        a:   min, mean, max [0.2, 0.5, 2]
        b:   min, mean, max [0.5, 8/(π**2), 2]
        bp:  8-bit offset in x, range [-inf, 255]
        bp2: 8-bit offset in y, range [-inf, 255]"""
        self.params = [np.array(p, dtype=np.float32) for p in (a, b, bp / 255, bp2 / 255)]
        self.inverse = bool(inverse)
        self.x_min = 1e-6
        self.x_max = 1.0 - 1e-6

    def __call__(self, x):
        assert (x.shape[0] in {1, 3}) and (x.ndim > 1), f'x has wrong shape: {x.shape}, expecting (C, *)'
        a, b, bp, bp2 = (p[:, None] for p in self.params)

        # limit x to ensure positive x and cos
        x = np.clip(bp + x * (1 - bp), self.x_min, self.x_max)

        if self.inverse:
            y = π_half**(-1 / a) * np.arccos((1 - x)**(1 / b))**(1 / a) * (1 - bp2) + bp2
        else:
            y = (1 - np.cos(π_half * x ** a) ** b) * (1 - bp2) + bp2
        return y


class InvBetaDataset(Dataset):
    """Images are transformed according to randomly drawn curve parameters"""

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
        self.dist_a = torch.distributions.normal.Normal(0, 0.5)
        self.dist_b = torch.distributions.normal.Normal(0.4, 0.25)
        self.dist_bp = torch.distributions.half_normal.HalfNormal(0.02)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        fn = os.path.join(self.image_root, self.df.iloc[index, 0])
        if 'gcsfs' in globals() and gcsfs.is_gcs_path(fn):
            bytes_data = gcsfs.read(fn)
            image = PIL.Image.open(io.BytesIO(bytes_data))
        else:
            # Benchmark (JPEGs): io 23:57, cv2 22:14, PIL 22:57
            # cv2 is slightly faster but does not exactly reproduce PIL/fastai's pixel values.
            #image = io.imread(fn)    # array
            assert os.path.exists(fn), f'{fn} not found'
            image = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
            #image = PIL.Image.fromarray(image)  # optional, not required
            #image = torch.from_numpy(cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)).permute(2,0,1).contiguous()/255.
            #image = torch.FloatTensor(cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)).to(device) / 255
            #image = PIL.Image.open(fn)

        n_channels = image.shape[-1]
        assert n_channels in {1, 3}, f'wrong image shape: {image.shape}, expecting channel last'
        
        labels = torch.stack([dist.sample((n_channels,)) for dist in [self.dist_a, self.dist_b, self.dist_bp]])
        assert labels.shape == (3, n_channels)

        for channel, label in enumerate(labels.T):
            curve = Curve(*label)
            image[:, :, channel] = curve.inverse_pdf(image[:, :, channel])

        labels = labels.flatten()

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


class InvGammaDataset(Dataset):
    """Images are transformed according to randomly drawn curve parameters"""

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
        self.dist_log_gamma = torch.distributions.normal.Normal(0, 0.4)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        fn = os.path.join(self.image_root, self.df.iloc[index, 0])
        if 'gcsfs' in globals() and gcsfs.is_gcs_path(fn):
            bytes_data = gcsfs.read(fn)
            image = PIL.Image.open(io.BytesIO(bytes_data))
        else:
            # Benchmark (JPEGs): io 23:57, cv2 22:14, PIL 22:57
            # cv2 is slightly faster but does not exactly reproduce PIL/fastai's pixel values.
            #image = io.imread(fn)    # array
            assert os.path.exists(fn), f'{fn} not found'
            image = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
            #image = PIL.Image.fromarray(image)  # optional, not required
            #image = torch.from_numpy(cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)).permute(2,0,1).contiguous()/255.
            #image = torch.FloatTensor(cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)).to(device) / 255
            #image = PIL.Image.open(fn)

        if self.transform:
            if self.albu:
                image = self.transform(image=np.array(image))['image']
                image = (image / 255).float() if self.floatify else image
            else:
                # torchvision, requires PIL.Image
                image = self.transform(PIL.Image.fromarray(image))

        n_channels = image.shape[0]
        assert n_channels in {1, 3}, f'wrong image shape: {image.shape}, expecting channel first'
        
        if self.tensor_transform:
            image = self.tensor_transform(image)

        # draw gamma (label), gamma-transform image
        labels = torch.exp(self.dist_log_gamma.sample((n_channels,)))
        image = torch.pow(image, labels[:, None, None])  # channel first

        return image, labels if self.labeled else image


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


class AugInvCurveDataset3(Dataset):
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
        if self.use_batch_tfms:
            self.presize = TT.Resize([int(s * cfg.presize) for s in cfg.size], 
                                     interpolation=InterpolationMode.NEAREST, antialias=cfg.antialias)
        else:
            #self.resize = TT.Resize(cfg.size, 
            #                        interpolation=InterpolationMode.BILINEAR, antialias=cfg.antialias)
            self.resize = TT.Resize(cfg.size, 
                                    interpolation=InterpolationMode.NEAREST, antialias=cfg.antialias)
        self.rng = np.random.default_rng()
        self.dist_log_gamma = partial(self.rng.uniform, *cfg.log_gamma_range)
        self.dist_curve3_a = partial(self.rng.uniform, *cfg.curve3_a_range)
        self.dist_curve3_beta = partial(self.rng.uniform, *cfg.curve3_beta_range)
        self.dist_curve4_loga = partial(self.rng.uniform, *cfg.curve4_loga_range)
        self.dist_curve4_b = partial(self.rng.uniform, *cfg.curve4_b_range)
        self.dist_bp = partial(self.rng.uniform, *cfg.blackpoint_range)
        self.dist_bp2 = partial(self.rng.uniform, *cfg.blackpoint2_range)
        self.p_gamma = cfg.p_gamma  # probability to use gamma (Curve0)
        self.p_beta = cfg.p_beta    # probability for Beta PDF (Curve3) rather than Curve4
        self.noise_level = cfg.noise_level
        self.add_uniform_noise = cfg.add_uniform_noise
        self.add_jpeg_artifacts = cfg.add_jpeg_artifacts
        self.sharpness_augment = cfg.sharpness_augment
        self.predict_inverse = cfg.predict_inverse
        self.mirror_gamma = cfg.mirror_gamma
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
        bp = self.dist_bp(n_channels).astype(np.float32)
        bp2 = self.dist_bp2(n_channels).astype(np.float32)
        curves = []
        
        # gamma
        log_gamma = self.dist_log_gamma(n_channels).astype(np.float32)
        gamma = np.exp(log_gamma)
        curves.append(Curve0(gamma, bp, bp2))

        # beta
        alpha = np.exp(self.dist_curve3_a(n_channels).astype(np.float32))
        beta = self.dist_curve3_beta(n_channels).astype(np.float32)
        curves.append(Curve3(alpha, beta, bp, bp2))

        # curve4
        a = np.exp(self.dist_curve4_loga(n_channels).astype(np.float32))
        b = self.dist_curve4_b(n_channels).astype(np.float32)
        curves.append(Curve4(a, b, bp, bp2))

        if self.curve_selection == 'channel-wise':
            targets, tfms = [], []
            for channel in range(n_channels):
                curve = curves[np.random.randint(0, 3)]
                tfm = curve.inverse(support[None, :])[channel]
                target = curve(support[None, :])[channel] if self.predict_inverse else tfm

                # random swap tfm <-> target
                if (self.predict_inverse and (np.random.random_sample() < 0.5) and any([
                (curve.__class__.__name__ == 'Curve0') and self.mirror_gamma,
                (curve.__class__.__name__ == 'Curve3') and self.mirror_beta,
                (curve.__class__.__name__ == 'Curve4') and self.mirror_curve4])):
                    tfm, target = target, tfm

                targets.append(target)
                tfms.append(tfm)

            target = np.stack(targets)
            tfm = np.stack(tfms)
            del targets, tfms

        else:
            # image-wise curve selection
            curve = curves[np.random.randint(0, 3)]
            tfm = curve.inverse(support[None, :])
            target = curve(support[None, :]) if self.predict_inverse else tfm

            # random swap tfm <-> target
            if (self.predict_inverse and (np.random.random_sample() < 0.5) and any([
                (curve.__class__.__name__ == 'Curve0') and self.mirror_gamma,
                (curve.__class__.__name__ == 'Curve3') and self.mirror_beta,
                (curve.__class__.__name__ == 'Curve4') and self.mirror_curve4])):
                mask = np.random.randint(0, 2, (n_channels, 1)).astype(np.float32)
                target, tfm = mask * target + (1 - mask) * tfm, (1 - mask) * target + mask * tfm
        
        target = torch.tensor(target)
        assert target.shape == (n_channels, 256), f"wrong target shape: {target.shape}"
        assert target.dtype == torch.float32, f"wrong target dtype: {target.dtype}"

        if self.use_batch_tfms:
            tfm = torch.tensor(tfm)

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

            return image, target

        image = map_index_torch(image, tfm, self.add_uniform_noise)
        image = (image.numpy().clip(0, 1) * 255).astype(np.uint8)

        if self.sharpness_augment:
            # randomly soften/sharpen the image
            rnd_sharpness = 1.8 * np.random.rand() + 0.1
            image = adjust_sharpness_alb(image, rnd_sharpness)

        if self.add_jpeg_artifacts:
            # adjust_jpeg_quality automatically converts image to uint8 and back
            rnd_quality = int(50 * (1 + np.random.rand()))
            image = adjust_jpeg_quality_tvf(image, rnd_quality)

        image = image.astype(np.float32) / 255

        # append rnd_factor for noise_level to target -> (C, 257)
        if self.noise_level:
            rnd_factor = torch.rand(1).repeat(3)
            target = torch.cat((target, rnd_factor[:, None]), dim=1)
            
        image = torch.tensor(image).permute(2, 0, 1)

        image = self.resize(image)

        return image, target


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
              add_uniform_noise=False, add_jpeg_artifacts=True, sharpness_augment=True):
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

    return image


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
    
    if cfg.curve is None:
        return tf.cast(image, tf.float32) / 255.0

    elif cfg.curve == 'beta':
        curves = get_curves(tfm, alpha_scale=cfg.alpha_scale or 1, beta_decay=cfg.beta_decay or 10)  # (256, C)
        image = map_index(image, curves, cfg.add_uniform_noise, height, width)

    elif cfg.curve in ['gamma', 'free']:
        curves = tf.transpose(tfm)  # (256, C)
        image = map_index(image, curves, height, width, 3,
                          cfg.add_uniform_noise, cfg.add_jpeg_artifacts, cfg.sharpness_augment)

    if cfg.curve not in ['gamma', 'free']:
        image /= 255.0

    if cfg.noise_level:
        rnd_factor = tf.random.uniform(())
        image += cfg.noise_level * rnd_factor * tf.random.normal((height, width, 3))
    
    image = tf.clip_by_value(image, 0, 1)  # does it matter?

    return image


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

        #if tf.random.uniform([]) < cfg.p_gamma:
        log_gamma = tf.random.uniform([3], *cfg.log_gamma_range)
        gamma = tf.exp(log_gamma)[:, None]

        if cfg.predict_inverse:
            x = support[None, :]
            x = bp + x * (1 - bp)
            x = tf.clip_by_value(x, 1e-6, 1)  # avoid nan
            x = tf.pow(x, gamma)
            x = x * (1 - bp2) + bp2
            target0 = tf.clip_by_value(x, 0, 1)

        x = support[None, :]
        x = (x - bp2) / (1 - bp2)
        x = tf.clip_by_value(x, 1e-6, 1)  # avoid nan
        x = tf.pow(x, 1 / gamma)
        x = (x - bp) / (1 - bp)
        tfm0 = tf.clip_by_value(x, 0, 1)

        if not cfg.predict_inverse:
            target0 = tfm0
        elif cfg.mirror_gamma and (tf.random.uniform([]) < 0.5):
            # randomly swap tfm/target
            target0, tfm0 = tfm0, target0

        #elif tf.random.uniform([]) < cfg.p_beta:
        a = tf.random.uniform([3], *cfg.curve3_a_range)
        alpha = tf.exp(a)[:, None]
        beta = tf.random.uniform([3], *cfg.curve3_beta_range)[:, None]

        if cfg.predict_inverse:
            x = support[None, :]
            x = bp + x * (1 - bp)
            x = tf.clip_by_value(x, 1e-6, 1 - 1e-6)  # avoid nan
            x = x * 0.9                              # reduce slope at whitepoint
            x = tf.pow(x, alpha - 1) * tf.pow(1 - x, beta - 1)  # unnormalized PDF(x)
            x /= x[:, -1:]                                      # normalize
            x = x * (1 - bp2) + bp2
            target1 = tf.clip_by_value(x, 0, 1)

        # float64 avoids div-by-zero below
        x = tf.cast(support, tf.float64)
        y = tf.linspace(1e-6, 1 - 1e-6, 2000)
        y = tf.cast(y, tf.float64)
        bp_64 = tf.cast(bp, tf.float64)
        bp2_64 = tf.cast(bp2, tf.float64)
        alpha = tf.cast(alpha, tf.float64)
        beta = tf.cast(beta, tf.float64)
        pdfs = bp_64 + y[None, :] * (1 - bp_64)
        pdfs = tf.clip_by_value(pdfs, 1e-6, 1 - 1e-6)  # avoid nan
        pdfs = pdfs * 0.9                              # reduce slope at whitepoint
        pdfs = tf.pow(pdfs, alpha - 1) * tf.pow(1 - pdfs, beta - 1)  # unnormalized PDF(x)
        pdfs /= pdfs[:, -1:]                                         # normalize
        pdfs = pdfs * (1 - bp2_64) + bp2_64
        pdfs = tf.clip_by_value(pdfs, 0, 1)
        tfm1 = tf.stack([interp1d_tf(pdfs[i], y, x) for i in range(3)])
        tfm1 = tf.cast(tf.clip_by_value(tfm1, 0, 1), tf.float32)

        if not cfg.predict_inverse:
            target1 = tfm1
        elif cfg.mirror_beta and (tf.random.uniform([]) < 0.5):
            # randomly swap tfm/target                
            target1, tfm1 = tfm1, target1

        #else:
        a = tf.exp(tf.random.uniform([3], *cfg.curve4_loga_range))[:, None]
        b = tf.random.uniform([3], *cfg.curve4_b_range)[:, None]
        π_half = 0.5 * tf.constant(np.pi)

        if cfg.predict_inverse:
            x = support[None, :]
            x = bp + x * (1 - bp)
            x = tf.clip_by_value(x, 1e-6, 1 - 1e-6)  # avoid nan
            x = 1 - tf.cos(π_half * x ** a) ** b
            x = x * (1 - bp2) + bp2
            target2 = tf.clip_by_value(x, 0, 1)

        x = support[None, :]
        x = (x - bp2) / (1 - bp2)
        x = tf.clip_by_value(x, 1e-6, 1 - 1e-6)  # avoid nan
        x = π_half**(-1 / a) * tf.math.acos((1 - x)**(1 / b))**(1 / a)
        x = (x - bp) / (1 - bp)
        tfm2 = tf.clip_by_value(x, 0, 1)

        if not cfg.predict_inverse:
            target2 = tfm2
        elif cfg.mirror_curve4:
            # randomly swap tfm/target channel-wise
            mask = tf.cast(tf.random.uniform([3, 1], minval=0, maxval=2, dtype=tf.int32), tf.float32)
            target2, tfm2 = mask * target2 + (1 - mask) * tfm2, (1 - mask) * target2 + mask * tfm2

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
        assert target.shape == (3 * 256,), f"wrong target shape: {target.shape}"
        assert target.dtype == tf.float32, f"wrong target dtype: {target.dtype}"

        #tf.print("tfm:", tfm.shape, tf.reduce_min(tfm), tf.reduce_mean(tfm), tf.reduce_max(tfm))  # tfm: TensorShape([3, 256]) 0 0.598787069 1

    features['height'] = example[cfg.data_format['height']]
    features['width'] = example[cfg.data_format['width']]
    features['image'] = decode_image(cfg, example[cfg.data_format['image']], tfm,
                                     features['height'], features['width'])

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

    # tf.keras.model.fit() wants dataset to yield a tuple (inputs, targets, [sample_weights])
    # inputs can be a dict
    inputs = tuple(features[key] for key in cfg.inputs)
    targets = tuple(features[key] for key in cfg.targets)
    #print("inputs:", type(inputs), inputs)  # <class 'tuple'> (<tf.Tensor 'clip_by_value_2:0' shape=(None, None, 3) dtype=float32>,)
    #print("targets", type(targets), targets)  # <class 'tuple'> (<tf.Tensor 'Const_1:0' shape=(768,) dtype=float32>,)

    return inputs, targets

def count_examples_in_tfrec(fn):
    count = 0
    for _ in tf.data.TFRecordDataset(fn):
        count += 1
    return count

def count_data_items(filenames, tfrec_filename_pattern=None):
    if '-of-' in filenames[0]:
        # Imagenet: Got number of items from idx files
        return np.sum([391 if 'validation' in fn else 1252 for fn in filenames])
    if 'coco' in filenames[0]:
        return np.sum([159 if 'train/coco184' in fn else 499 if 'val/coco7' in fn else 
                       642 if '/train/' in fn else 643 
                       for fn in filenames])
        #print(f'{len(filenames)} urls:')
        #total_count = 0
        #for fn in filenames:
        #    n = count_examples_in_tfrec(fn)
        #    print("   ", Path(fn).parent, n)
        #    total_count += n
        #return total_count
    raise NotImplementedError(f'autolevels.count_data_items: filename not recognized: {filenames[0]}')


class TFCurveRMSE(tf.keras.metrics.MeanSquaredError):
    def __init__(self, name='curve_rmse', curve='gamma', **kwargs):
        if curve == 'gamma': name = 'rmse'
        super().__init__(name=name, **kwargs)
        self.curve = curve

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
