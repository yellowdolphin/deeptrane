import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as TT
from torchvision.transforms.functional import InterpolationMode
from scipy.interpolate import interp1d


try:
    import albumentations as alb
except ModuleNotFoundError:
    from utils.general import quietly_run
    quietly_run('pip install albumentations')
    import albumentations as alb
print("[ √ ] albumentations:", alb.__version__)


import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

gcs_paths = {
    'imagenet-object-localization-challenge': 'gs://kds-ea17055740cb6a131247a93dc6ea8c461c783584c675b3a2391fbddb',
    }

def init(cfg):
    cfg.competition_path = Path('/kaggle/input/imagenet-object-localization-challenge')
    if cfg.cloud == 'drive':
        cfg.competition_path = Path(f'/content/gdrive/MyDrive/{cfg.project}')

    if cfg.filetype == 'JPEG':
        cfg.image_root = cfg.competition_path / 'ILSVRC/Data/CLS-LOC/train'
    else:
        # Customize data pipeline (see tf_data for definition and defaults)
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
        #cfg.tfrec_format = {'encoded': tf.io.FixedLenFeature([], tf.string)}
        cfg.tfrec_format = {'image/encoded': tf.io.FixedLenFeature([], tf.string),
                            'image/height': tf.io.FixedLenFeature([], tf.int64),
                            'image/width': tf.io.FixedLenFeature([], tf.int64)}
        cfg.data_format = {'image': 'image/encoded',
                           'height': 'image/height',
                           'width': 'image/width'}
        cfg.inputs = ['image']
        cfg.targets = ['target']

    cfg.meta_csv = cfg.competition_path / 'ILSVRC/ImageSets/CLS-LOC/train_cls.txt'

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
    elif cfg.curve == 'beta':
        cfg.dataset_class = AugInvBetaDataset
        #cfg.channel_size = 9  # set in config file

def read_csv(cfg):
    return pd.read_csv(cfg.meta_csv, sep=' ', usecols=[0], header=None, names=['image_id'])


class OptimizableCurve(torch.nn.Module):
    def __init__(self, a=1, b=1, bp=0, eps=0.1):
        """Differentiable probability density function of a Beta distribution

        a: log(alpha parameter)
        b: logit(beta parameter)
        bp: black point / 500
        eps: small positive (nonzero) value, increase to reduce peak slope"""
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(float(a)))
        self.b = torch.nn.Parameter(torch.tensor(float(b)))
        self.bp = torch.nn.Parameter(torch.tensor(float(bp)))
        self.eps = float(eps)
        self.max_x = torch.tensor(1 - self.eps, dtype=torch.float32)

    def forward(self, x):
        "Input tensor value range: 0 ... 1"
        x = torch.clip(x, min=1e-6, max=1) * self.max_x
        alpha, beta, blackpoint = self.transformed_parameters()
        dist = Beta(alpha, beta)
        c = (255. - blackpoint) / torch.exp(dist.log_prob(self.max_x))
        return blackpoint + c * torch.exp(dist.log_prob(x))
    
    def pdf(self, x):
        # TEST: is this slower than forward?
        "Input array value range: 0 ... 255"
        alpha, beta, blackpoint = self.transformed_parameters()
        x = np.clip(x, a_min=1e-3, a_max=None) / 255 * (1 - self.eps)
        y = np.power(x, alpha - 1) * np.power(1 - x, beta - 1)
        return blackpoint + (255 - blackpoint) * y / y.max()

    def transformed_parameters(self):
        alpha = 1 + torch.exp(self.a.detach())
        beta = torch.sigmoid(10 * self.b.detach())
        blackpoint = 500 * self.bp.detach()
        return alpha, beta, blackpoint

    @classmethod
    def from_transformed_parameters(cls, alpha, beta, blackpoint, eps=0.1):
        a = np.log(alpha - 1)
        b = torch.logit(torch.tensor(beta)) / 10
        bp = blackpoint / 500
        return cls(a, b, bp, eps)

    def inverse_pdf(self, x):
        "Input x (array-like): range 0...255"
        alpha, beta, blackpoint = self.transformed_parameters()
        assert x.max() > 1, f"inv_pdf called with x={x.dtype} {x.shape} {x.min()} ... {x.max()}"
        df = pd.DataFrame({'y': np.linspace(0, 255, 10000)})
        df['x'] = self.pdf(df.y.to_numpy())
        df.x = df.x.astype(int)  # pitfall: np.uint64 + int -> np.float64
        m = df.groupby('x').y.mean()  # sorted int index
        if len(m) < 1 + m.index[-1] - m.index[0]:
            #print("using interpolation")
            m = m.reindex(range(m.index[0], m.index[-1] + 1)).interpolate('linear')
        assert len(m) == 1 + m.index.max() - m.index.min(), f"m.index: {m.index.min()} ... {m.index.max()} ({len(m)})"
        shape = x.shape
        x = np.clip(x, a_min=m.index[0], a_max=m.index[-1])
        x = x.astype(np.uint64).reshape(-1)
        return m.loc[x].to_numpy().reshape(shape)


class Curve():
    def __init__(self, a=1, b=1, bp=0, eps=0.1, create_blackpoint=False):
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
        return blackpoint + (255 - blackpoint) * y / y.max()

    def transformed_parameters(self):
        alpha = 1 + np.exp(self.a)
        beta = 1 / (1 + np.exp(-(10 * self.b)))  # sigmoid
        blackpoint = 500 * self.bp
        return alpha, beta, blackpoint

    @classmethod
    def from_transformed_parameters(cls, alpha, beta, blackpoint, eps=0.1):
        a = np.log(alpha - 1)
        b = np.log(beta / (1 - beta)) / 10  # logit
        bp = blackpoint / 500
        return cls(a, b, bp, eps)

    def get_inverse_curves(self):
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
        return np.stack(curves)  # channels first
    
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
            self.presize = TT.Resize([s * 2 for s in cfg.size], interpolation=InterpolationMode.NEAREST)
        self.dist_log_gamma = torch.distributions.normal.Normal(0, 0.4)
        self.noise_level = cfg.noise_level

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

        # draw gamma (label)
        labels = torch.exp(self.dist_log_gamma.sample((n_channels,)))

        if self.use_batch_tfms:
            # just resize to double cfg.size and tensorize for collocation
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
        self.dist_a = torch.distributions.normal.Normal(0, 0.5)
        self.dist_b = torch.distributions.normal.Normal(0.4, 0.25)
        self.dist_bp = torch.distributions.half_normal.HalfNormal(0.02)
        self.use_batch_tfms = cfg.use_batch_tfms
        if self.use_batch_tfms:
            self.presize = TT.Resize([s * 2 for s in cfg.size], interpolation=InterpolationMode.NEAREST)
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
        curves = Curve(*tuple(p.numpy() for p in labels), create_blackpoint=True)
        labels = torch.stack(labels)
        labels = labels.transpose(0, 1)  # channel first
        assert labels.shape == (n_channels, 3)

        if self.use_batch_tfms:
            # just resize to double cfg.size and tensorize for collocation
            if self.curve_tfm_on_device:
                image = torch.tensor(image.transpose(2, 0, 1))  # channel first
                image = self.presize(image)
                curves = torch.tensor(curves.get_inverse_curves(), dtype=torch.float32)
                assert curves.shape == (n_channels, 256)
                assert curves.dtype == labels.dtype, f'{curves.dtype} != {labels.dtype}'
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


# TF data pipeline --------------------------------------------------------

def get_curves(target):
    # target: (2, C) for gamma + bp, (3, C) for a, b, bp (beta-curve)
    # curves: (256, C)
    gamma, bp = target[0, :], target[1, :]  # iter over symbolic tensor not allowed!
    curves = tf.range(0., 256., delta=1., dtype=tf.float32)[:, None]
    curves = tf.math.pow(curves, gamma[None, :])
    curves = curves * ((255. - bp[None, :]) / 255.) + bp[None, :]
    return curves


def curve_tfm(image, target):
    # image: channel last
    # target: (2, C) for gamma + bp, (3, C) for a, b, bp (beta-curve)
    gamma, bp = target[0, :], target[1, :]  # iter over symbolic tensor not allowed!
    bp = bp[None, None, :] / 255.
    image = image * (1. - bp) + bp
    image = tf.math.pow(image, gamma[None, None, :])
    return image


@tf.function
def map_index(image, curves):
    # OperatorNotAllowedInGraphError: Iterating over a symbolic `tf.Tensor` is not allowed
    #for i, curve in enumerate(curves):
    #    image[:, :, i] = curve[image[:, :, i]]
    # No fancy indexing in TF:
    #for i in range(3):
    #    image[:, :, i] = curves[:, i][image[:, :, i]]
    # Use tf.gather_nd instead:
    image = tf.cast(image, tf.int32)
    curves = tf.cast(curves, tf.float32)
    assert curves.shape[-1] in {1, 3}
    channel_idx = tf.ones_like(image, dtype=tf.int32) * tf.range(3)[None, None, :]
    indices = tf.stack([image, channel_idx], axis=-1)
    return tf.gather_nd(curves, indices)


def decode_image(cfg, image_data, target, height, width):
    "Decode image and apply inverse curve transform according to target"
    
    image = tf.image.decode_jpeg(image_data, channels=3)
    
    if cfg.BGR:
        image = tf.reverse(image, axis=[-1])

    if cfg.normalize in ['torch', 'tf', 'caffe']:
        from keras.applications.imagenet_utils import preprocess_input
        image = preprocess_input(image, mode=cfg.normalize)
    elif cfg.curve == 'beta':
        # (a) try index mapping
        image = tf.cast(image, tf.int32)  # tf.gather_nd needs int32
        curves = get_curves(target)  # (256, C)
        image = map_index(image, curves)
        # Cannot use image.shape: ValueError: Cannot convert a partially known TensorShape (None, None, 3) to a Tensor
        if cfg.noise_level:
            image += tf.random.normal((height, width, 3)) * cfg.noise_level
        image = tf.clip_by_value(image, clip_value_min=0., clip_value_max=255.)
        image /= 255.0

        # (b) try curve transform the image
        #image = tf.cast(image, tf.float32)
        #image = image + tf.random.uniform((height, width, 3))
        #image /= 255.0
        #image = curve_tfm(image, target)
        #image = tf.clip_by_value(image, clip_value_min=0., clip_value_max=1.)

    elif cfg.curve == 'gamma':
        image = tf.cast(image, tf.float32)
        image = image + tf.random.uniform((height, width, 3))
        image = tf.clip_by_value(image, clip_value_min=0., clip_value_max=255.)
        image /= 255.0
        image = tf.math.pow(image, target[None, None, :])
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

    if cfg.curve == 'gamma':
        features['target'] = tf.math.exp(tfd.Normal(0.0, 0.4).sample([3]))
    elif cfg.curve == 'beta':
        # target: (2, C) for gamma + bp, (3, C) for a, b, bp (beta-curve)
        gamma = tf.math.exp(tfd.Normal(0.0, 0.4).sample([3]))
        bp = tfd.HalfNormal(scale=40.).sample([3])
        features['target'] = tf.stack([gamma, bp])

    features['height'] = example[cfg.data_format['height']]
    features['width'] = example[cfg.data_format['width']]
    features['image'] = decode_image(cfg, example[cfg.data_format['image']], features['target'],
                                     features['height'], features['width'])
    
    # tf.keras.model.fit() wants dataset to yield a tuple (inputs, targets, [sample_weights])
    # inputs can be a dict
    inputs = tuple(features[key] for key in cfg.inputs)
    targets = tuple(features[key] for key in cfg.targets)

    return inputs, targets


def count_data_items(filenames, tfrec_filename_pattern=None):
    "Got number of items from idx files"
    return np.sum([391 if 'validation' in fn else 1252 for fn in filenames])
