import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
print("import torch...")
import torch
from torch.utils.data import Dataset
import torchvision
print("[ √ ] torchvision:", torchvision.__version__)
import torchvision.transforms as TT
from torchvision.transforms.functional import InterpolationMode
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
        cfg.targets = ['target_rel', 'target_abs']
    elif cfg.curve == 'beta':
        cfg.dataset_class = AugInvBetaDataset
        assert cfg.channel_size, "Set channel_size in config file!"
        cfg.targets = (['target_gamma', 'target_bp'] if cfg.channel_size == 6 else
                       ['target_a', 'target_b', 'target_bp'])
    else:
        cfg.dataset_class = AugInvCurveDataset
        cfg.channel_size = 256 * 3
        cfg.targets = ['target']


def read_csv(cfg):
    if cfg.DEBUG:
        return pd.read_csv(cfg.meta_csv, sep=' ', usecols=[0], header=None, names=['image_id']).sample(frac=0.01)
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


class AugInvCurveDataset(Dataset):
    """Images are mapped on device using the randomly generated target_curve"""

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
            self.presize = TT.Resize([s * 2 for s in cfg.size], interpolation=InterpolationMode.NEAREST)
        #self.dist_log_gamma = torch.distributions.normal.Normal(0, 0.4)
        self.dist_log_gamma = torch.distributions.uniform.Uniform(*cfg.log_gamma_range) if cfg.log_gamma_range else None
        self.dist_a = torch.distributions.normal.Normal(0, cfg.a_sigma or 0.5) if cfg.a_sigma else None
        self.dist_b = torch.distributions.normal.Normal(cfg.b_mean or 0.4, cfg.b_sigma) if cfg.b_sigma else None
        self.dist_bp = torch.distributions.normal.Normal(0, cfg.bp_sigma / 255) if cfg.bp_sigma else None
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

        # Generate curve (C, 256)
        curve = torch.linspace(0, 1, 256, dtype=torch.float32)
        if self.dist_log_gamma:
            log_gamma = self.dist_log_gamma.sample((n_channels,))
            gamma = torch.exp(log_gamma)
            curve = torch.pow(curve[None, :], gamma[:, None])
        elif self.dist_a and self.dist_b:
            a = self.dist_a.sample((n_channels,))
            b = self.dist_b.sample((n_channels,))
            curve = Curve(a, b).get_inverse_curves()
            assert curve.shape == (n_channels, 256), f"wrong curve shape: {curve.shape}"
            assert curve.dtype == torch.float32, f"wrong curve dtype: {curve.dtype}"
        if self.dist_bp:
            bp_shift = self.dist_bp.sample((n_channels,))
            curve = (curve + bp_shift[:, None]).clip(0, None) / (1 + bp_shift[:, None])

        assert self.use_batch_tfms, 'AugInvCurveDataset only supports batch_tfms'
        # append rnd_factor for noise_level -> (C, 257)
        if self.noise_level:
            rnd_factor = torch.rand(1).repeat(3)
            curve = torch.cat((curve, rnd_factor[:, None]), dim=1)

        # resize image to double cfg.size and tensorize for collocation
        image = torch.tensor(image.transpose(2, 0, 1))  # channel first
        image = self.presize(image)

        return image, curve


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


def map_index(image, curves, add_uniform_noise=True, height=None, width=None):
    # OperatorNotAllowedInGraphError: Iterating over a symbolic `tf.Tensor` is not allowed
    #for i, curve in enumerate(curves):
    #    image[:, :, i] = curve[image[:, :, i]]
    # No fancy indexing in TF:
    #for i in range(3):
    #    image[:, :, i] = curves[:, i][image[:, :, i]]
    # Use tf.gather_nd instead:
    image = tf.cast(image, tf.int32)  # tf.gather_nd needs int32
    if add_uniform_noise: image_plus_one = tf.clip_by_value(image + 1, 0, 255)
    curves = tf.cast(curves, tf.float32)
    #assert curves.shape[-1] in {1, 3}
    channel_idx = tf.ones_like(image, dtype=tf.int32) * tf.range(3)[None, None, :]
    indices = tf.stack([image, channel_idx], axis=-1)
    if add_uniform_noise: indices_plus_one = tf.stack([image_plus_one, channel_idx], axis=-1)
    image = tf.gather_nd(curves, indices)
    if add_uniform_noise: 
        image_plus_one = tf.gather_nd(curves, indices_plus_one)
        noise_range = image_plus_one - image
        noise = tf.random.uniform((height, width, 3)) * noise_range
        image += noise
    return image


def decode_image(cfg, image_data, target, height, width):
    "Decode image and apply inverse curve transform according to target"
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

    if cfg.curve == 'beta':
        curves = get_curves(target, alpha_scale=cfg.alpha_scale or 1, beta_decay=cfg.beta_decay or 10)  # (256, C)
        image = map_index(image, curves, cfg.add_uniform_noise, height, width)
        #curves = tf.reshape(curves, (-1,))

    elif cfg.curve == 'gamma':
        image = tf.cast(image, tf.float32)
        if cfg.add_uniform_noise:
            if (cfg.add_uniform_noise is True) or (tf.random.uniform([]) < cfg.add_uniform_noise):
                image += tf.random.uniform((height, width, 3))

    elif cfg.curve == 'free':
        curves = tf.transpose(target)
        image = map_index(image, curves, cfg.add_uniform_noise, height, width)

    if cfg.curve != 'free':
        image /= 255.0

    #tf.print("image:", image.shape, tf.reduce_min(image), tf.reduce_mean(image), tf.reduce_max(image))
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    if cfg.curve == 'gamma':
        image = tf.math.pow(image, target[None, None, :])

    if cfg.random_blackpoint_shift:
        bp_shift = cfg.random_blackpoint_shift / 255 * tf.random.normal((3,))
        image += bp_shift[None, None, :]
        image /= (1 + bp_shift[None, None, :])

    if cfg.noise_level:
        rnd_factor = tf.random.uniform(())
        image += cfg.noise_level * rnd_factor * tf.random.normal((height, width, 3))
    
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)  # does it matter?

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
        #features['target'] = tf.math.exp(tfd.Normal(0.0, 0.6).sample([3]))
        features['target'] = tf.math.exp(tfd.Uniform(low=-1.6, high=1.0).sample([3]))
    elif cfg.curve == 'beta' and cfg.channel_size == 2:
        # target: (2, C) for gamma + bp, (3, C) for a, b, bp (beta-curve)
        gamma = tf.math.exp(tfd.Normal(0.0, 0.4).sample([3]))
        bp = tfd.HalfNormal(scale=40.).sample([3])
        features['target'] = tf.stack([gamma, bp])
    elif cfg.curve == 'beta':
        # do not instantiate tfd classes globally: TF allocates half of GPU!
        dist_a = tfd.Normal(0.0, scale=cfg.a_sigma or 0.5)
        dist_b = tfd.Normal(cfg.b_mean or 0.4, scale=cfg.b_sigma or 0.25)
        dist_bp = tfd.HalfNormal(scale=cfg.bp_sigma or 0.02)
        a = dist_a.sample([3])
        b = dist_b.sample([3])
        bp = dist_bp.sample([3])
        features['target'] = tf.stack([a, b, bp])
    elif cfg.curve == 'free':
        dist_log_gamma = tfd.Uniform(*cfg.log_gamma_range) if cfg.log_gamma_range else None
        dist_a = tfd.Normal(0.0, scale=cfg.a_sigma or 0.5) if cfg.a_sigma else None
        dist_b = tfd.Normal(cfg.b_mean or 0.4, scale=cfg.b_sigma) if cfg.b_sigma else None
        dist_bp = tfd.Normal(0.0, cfg.bp_sigma / 255) if cfg.bp_sigma else None

        # Generate curve (C, 256)
        curve = tf.linspace(0.0, 1.0, 256)
        if dist_log_gamma:
            log_gamma = dist_log_gamma.sample([3])
            gamma = tf.math.exp(log_gamma)
            curve = tf.math.pow(curve[None, :], gamma[:, None])
        elif dist_a and dist_b:
            a = dist_a.sample([3])
            b = dist_b.sample([3])
            beta_decay = cfg.beta_decay or 10
            alpha = 1 + tf.exp(a)[:, None]
            beta = 1 / (1 + tf.exp(-(beta_decay * b)))[:, None]  # sigmoid
            y = tf.linspace(1e-3, 0.9, 2000)[None, :]
            xs = tf.pow(y, alpha - 1) * tf.pow(1 - y, beta - 1)  # pdf(y)
            q = tf.linspace(0.0, 255.0, 256)
            #q = tf.range(0.0, 256.0, delta=1.0, dtype=tf.float32)
            curve = tf.stack([interp1d_tf(xs[c, :], y, q) for c in range(3)])
            assert curve.shape == (3, 256), f"wrong curve shape: {curve.shape}"
            assert curve.dtype == tf.float32, f"wrong curve dtype: {curve.dtype}"
        if dist_bp:
            bp_shift = dist_bp.sample([3])
            #curve = tf.clip_by_value((curve + bp_shift[:, None]) / (1 + bp_shift[:, None]), 0, 1)
            curve = (curve + bp_shift[:, None]) / (1 + bp_shift[:, None])

        features['target'] = curve
        #tf.print("curve:", curve.shape, tf.reduce_min(curve), tf.reduce_mean(curve), tf.reduce_max(curve))

    features['height'] = example[cfg.data_format['height']]
    features['width'] = example[cfg.data_format['width']]
    features['image'] = decode_image(cfg, example[cfg.data_format['image']], features['target'],
                                     features['height'], features['width'])

    if cfg.curve == 'gamma':
        # predict relative log_gamma and absolute log_gamma
        log_gamma = tf.math.log(features['target'])
        relative_log_gamma = log_gamma - tf.math.reduce_mean(log_gamma, keepdims=True)
        features['target_rel'] = relative_log_gamma
        features['target_abs'] = log_gamma

    if cfg.curve == 'beta':
        # split target into 2 (3) components for weighted loss
        if cfg.channel_size == 6:
            features['target_gamma'] = features['target'][0, :]
            features['target_bp'] = features['target'][1, :]
        else:
            features['target_a'] = features['target'][0, :]
            features['target_b'] = features['target'][1, :]
            features['target_bp'] = features['target'][2, :]

    if cfg.curve == 'free':
        features['target'] = tf.reshape(features['target'], [cfg.channel_size])  # match model output

    # tf.keras.model.fit() wants dataset to yield a tuple (inputs, targets, [sample_weights])
    # inputs can be a dict
    inputs = tuple(features[key] for key in cfg.inputs)
    targets = tuple(features[key] for key in cfg.targets)

    return inputs, targets


def count_data_items(filenames, tfrec_filename_pattern=None):
    "Got number of items from idx files"
    return np.sum([391 if 'validation' in fn else 1252 for fn in filenames])


class TFCurveRMSE(tf.keras.metrics.MeanSquaredError):
    def __init__(self, name='curve_rmse', curve='gamma', **kwargs):
        super().__init__(name=name, **kwargs)
        self.curve = curve

    def update_state(self, y_true, y_pred, sample_weight=None):
        support = tf.cast(tf.linspace(0, 1, 256), dtype=self._dtype)
        if self.curve == 'gamma':
            # (256,) ** (N, 3) -> (N, 3, 256)
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
