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

π = np.pi
π_half = 0.5 * np.pi

def init(cfg):
    if cfg.filetype:
        cfg.image_root = (
            cfg.image_root if (cfg.cloud == 'drive') else
            Path('/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train') if 'imagenet' in cfg.tags else
            Path('/kaggle/input/coco-2017-dataset/coco2017') if 'coco2017' in cfg.tags else
            Path('/kaggle/input/landmark-recognition-2021') if 'landmark2021' in cfg.tags else
            Path('define image_root in project module!'))

    # New datasets for pytorch: Set meta_csv to None to search for images and generate metadata.csv on the fly 
    cfg.meta_csv = (
        cfg.meta_csv if (cfg.cloud == 'drive') else
        Path('/kaggle/input/imagenet-object-localization-challenge/ILSVRC/ImageSets/CLS-LOC/train_cls.txt') if 'imagenet' in cfg.tags else
        Path('/kaggle/input/autolevels-modelbox/coco2017.csv') if 'coco2017' in cfg.tags else
        Path('/kaggle/input/autolevels-modelbox/landmark2021.csv') if 'landmark2021' in cfg.tags else
        Path('define cfg.meta_csv in project module!'))

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
            
    # LayerNorm between global_pool and first FC layer
    if 'tiny_vit' in cfg.arch_name:
        # At least for this arch, it is better to remove it.
        cfg.replace_body_layers = {'head.norm': 'Identity'}


    if True:
        # rename layers to convert "features_only" model to "num_classes" model
        def translate(name, replacements):
            for old, new in replacements.items():
                name = name.replace(old, new)
            return name
        
        replacements = {
            'stages_': 'stages.',
            'head.2': 'body.head.fc',
            'head.4': 'head.2',
            'head.6': 'head.4',
            'head.8': 'head.6',
            'head.10': 'head.8'}
        
        def modify_state_dict(state_dict, pretrained_model_state_dict):
            if ('head.10.weight' in state_dict) and ('head.10.weight' not in pretrained_model_state_dict):
                from collections import OrderedDict
                print("Modifying loaded state_dict to match custom head...")
                return OrderedDict((translate(k, replacements), v) for k, v in state_dict.items())
            return state_dict
        
        cfg.modify_state_dict = modify_state_dict

    #cfg.improver = True  # set in config file if desired

    if cfg.improve_color_loss:
        color_weight = cfg.improve_color_loss
        from torch.nn.functional import mse_loss

        def improve_color_loss(preds, labels):
            # preds/labels reshape [N, 3 * 256] into [N, C, 256]
            residuals = (preds - labels).reshape(-1, 3, 256)

            residual_L = torch.mean(residuals, dim=1, keepdim=True)
            color_loss = 2.0 * torch.mean(torch.square(residuals - residual_L))

            return color_weight * color_loss + (1 - color_weight) * mse_loss(preds, labels)

        cfg.criterion = improve_color_loss
    else:
        cfg.criterion = torch.nn.MSELoss()


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


def adjust_sharpness_alb(image, sharpness):
    """Albumentation equivalent to torchvision.transforms.functional.adjust_sharpness.
    
    Parameters:
        image (np.array): Input image.
        sharpness (float): As the blur kernel differs in albumentation/torchvision, 
            a sharpness of 1.7 corresponds to a sharpness of 2.0 in torchvision.

    Returns:
        np.array: Output image (uint8).
    """
    image = image.astype(np.float32) * sharpness + blur(image, ksize=3).astype(np.float32) * (1 - sharpness)
    return image.clip(0, 255).astype(np.uint8)


def adjust_jpeg_quality_alb(img, quality):
    return ImageCompression(quality_lower=quality, quality_upper=quality, 
                            always_apply=True)(image=img)['image']


def adjust_jpeg_quality_tvf(image, quality):
    image = torch.tensor(image).permute(2, 0, 1)  # convert to CHW tensor
    image = decode_jpeg(encode_jpeg(image, quality))
    return image.permute(1, 2, 0).numpy()  # return HWC numpy array


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


class Curve0():
    def __init__(self, gamma=1.0, bp=0, bp2=0, wp=255, bp_clip=None, unclipped=False):
        """Function  y(x) = x^a  with offsets bp, bp2 in x, y

        Input/Output range: [0, 1]

        Parameters
        gamma: min, mean, max [0.25, 1, 3]
        bp:  8-bit offset in x, range [-inf, 255]
        bp2: 8-bit offset in y, range [-inf, 255]

        inverse() returns the inverse function 
            y^-1(x) = x^(1 / gamma)
        """
        self.params = [np.array(p, dtype=np.float32).reshape(-1) for p in (gamma, bp / 255, bp2 / 255, wp / 255)]
        self.x_min = 1e-6
        self.x_max = None
        self.bp_clip = max(int(bp_clip), 0) if bp_clip is not None else None
        self.bp_is_clipped = False
        self.unclipped = bool(unclipped)

    def __call__(self, x):
        assert (x.shape[0] in {1, 3}) and (x.ndim > 1), f'x has wrong shape: {x.shape}, expecting (C, *)'
        if not self.bp_is_clipped: _ = self.inverse(x)
        gamma, bp, bp2, wp = (p[:, None] for p in self.params)

        x = bp + x * ((1 - bp) / wp)
        x = np.clip(x, self.x_min, self.x_max)  # avoid nan
        x = np.power(x, gamma)
        x = x * (1 - bp2) + bp2
        return x if self.unclipped else np.clip(x, 0, 1)
    
    def inverse(self, x):
        assert (x.shape[0] in {1, 3}) and (x.ndim > 1), f'x has wrong shape: {x.shape}, expecting (C, *)'
        gamma, bp, bp2, wp = (p[:, None] for p in self.params)

        x = (x - bp2) / (1 - bp2)
        x = np.clip(x, self.x_min, self.x_max)  # avoid nan
        x = np.power(x, 1 / gamma)
        if self.bp_clip:
            assert x.shape == (3, 256), f'x has unexpected shape {x.shape}'
            bp_max = x[:, self.bp_clip:self.bp_clip + 1]
            bp = np.minimum(bp, bp_max)
            self.params[1] = bp[:, 0]  # propagate to future calls
            self.bp_is_clipped = True
        x = (x - bp) / ((1 - bp) / wp)
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
        assert np.min(beta) > 0, f'beta ({beta}) out of scope, must be > 0'
        #assert np.all(np.logical_or((beta < 1), (alpha > 1.6))), f'(alpha, beta) ({alpha}, {beta}) out of scope'
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
    def __init__(self, a=0.5, b=0.81, bp=0, bp2=0, bp_clip=None, wp=255, mirror_mask=None, unclipped=False):
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
        self.params = [np.array(p, dtype=np.float32).reshape(-1) for p in (a, b, bp / 255, bp2 / 255, wp / 255)]
        self.x_min = 1e-6
        self.x_max = 1.0 - 1e-6
        self.bp_clip = max(int(bp_clip), 0) if bp_clip is not None else None
        self.bp_is_clipped = False
        self.mirror_mask = np.ones((3, 1), dtype=np.float32) if mirror_mask is None else mirror_mask
        self.unclipped = bool(unclipped)

    def __call__(self, x):
        assert (x.shape[0] in {1, 3}) and (x.ndim > 1), f'x has wrong shape: {x.shape}, expecting (C, *)'
        if not self.bp_is_clipped: _ = self.inverse(x)
        a, b, bp, bp2, wp = (p[:, None] for p in self.params)

        x = bp + x * ((1 - bp) / wp)
        x = self.mirror_mask * self.curve4(x, a, b) + (1 - self.mirror_mask) * self.inv_curve4(x, a, b)
        x = x * (1 - bp2) + bp2
        return x if self.unclipped else np.clip(x, 0, 1)
    
    def inverse(self, x):
        assert (x.shape[0] in {1, 3}) and (x.ndim > 1), f'x has wrong shape: {x.shape}, expecting (C, *)'
        a, b, bp, bp2, wp = (p[:, None] for p in self.params)

        x = (x - bp2) / (1 - bp2)
        x = self.mirror_mask * self.inv_curve4(x, a, b) + (1 - self.mirror_mask) * self.curve4(x, a, b)
        if self.bp_clip:
            assert x.shape == (3, 256), f'x has unexpected shape {x.shape}'
            bp_max = x[:, self.bp_clip:self.bp_clip + 1]
            bp = np.minimum(bp, bp_max)
            self.params[2] = bp[:, 0]  # propagate to future calls
            self.bp_is_clipped = True
        x = (x - bp) / ((1 - bp) / wp)
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
                 return_path_attr=None, param_csv_file=None):
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
        if cfg.curve3_conditional_a_range is None:
            self.curve3_b_weight = 0
            self.curve3_a_range = cfg.curve3_a_range
            self.curve3_alpha_0 = 0
        elif 'conditional01' in cfg.tags:
            self.curve3_b_weight = cfg.curve3_conditional_weight or 0.6
            self.curve3_a_range = cfg.curve3_conditional_a_range
            self.curve3_alpha_0 = 0
        else:
            self.curve3_b_weight = cfg.curve3_conditional_weight or 0.6
            self.curve3_a_range = cfg.curve3_conditional_a_range
            self.curve3_alpha_0 = 1
        self.curve3_beta_range = cfg.curve3_beta_range
        self.curve4_loga_range = cfg.curve4_loga_range
        self.curve4_conditional_logb_offsets = cfg.curve4_conditional_logb_offsets
        if cfg.curve4_conditional_logb_range is None:
            self.curve4_loga_weight = None
            self.curve4_conditional_logb_offsets = None
            self.curve4_b_range = cfg.curve4_b_range
        elif 'conditional01' in cfg.tags:
            # switch from b to logb and make it dependent on loga
            self.curve4_loga_weight = cfg.curve4_conditional_weight or 0.6
            self.curve4_logb_range = cfg.curve4_conditional_logb_range
        else:
            # logb = logb_0 + logb_range * (loga - loga_0)
            assert self.curve4_conditional_logb_offsets is not None, 'conditional02+ need logb_offsets'
            self.curve4_logb_range = cfg.curve4_conditional_logb_range
        self.bp_range = cfg.blackpoint_range
        self.bp2_range = cfg.blackpoint2_range
        self.bp_clip = max(*cfg.blackpoint_range, *cfg.blackpoint2_range) if cfg.clip_target_blackpoint else None
        self.wp_sigma = cfg.whitepoint_sigma or 0
        assert self.wp_sigma < 255, 'whitepoint_sigma must be smaller than 255'
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
        self.DEBUG = cfg.DEBUG
        self.param_csv_file = param_csv_file

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        fn = os.path.join(self.image_root, self.df.iloc[index, 0])
        if fn.startswith('virtual'):
            # Ignore file name from self.df, use dummy image
            image = np.empty((16, 16, 3), dtype='uint8')
        elif 'gcsfs' in globals() and gcsfs.is_gcs_path(fn):
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
        if self.wp_sigma:
            wp = 255 + 0.5 * self.wp_sigma * np.random.randn(n_channels)
            wp = np.clip(wp, 255 - self.wp_sigma, 255)  # truncate at sigma
        else:
            wp = 255
        curves = []

        # gamma
        log_gamma = np.random.uniform(*self.log_gamma_range, n_channels).astype(np.float32)
        gamma = np.exp(log_gamma)
        curves.append(Curve0(gamma, bp, bp2, wp, self.bp_clip))

        # beta
        beta = np.random.uniform(*self.curve3_beta_range, n_channels).astype(np.float32)
        a = np.random.uniform(*self.curve3_a_range, n_channels).astype(np.float32) + self.curve3_b_weight * beta
        #a = np.where(bp > 10.0, a.clip(0.5), a)  # lower limit on a if bp > 10 (not enough to avoid too steep curves)
        alpha = np.exp(a) + self.curve3_alpha_0
        mirror_mask = np.random.randint(low=0, high=2, size=(3, 1)).astype(np.float32) if self.mirror_beta else None
        curves.append(Curve3(alpha, beta, bp, bp2, self.bp_clip, mirror_mask))

        # curve4
        loga = np.random.uniform(*self.curve4_loga_range, n_channels).astype(np.float32)
        a = np.exp(loga)
        if self.curve4_conditional_logb_offsets is not None:
            # conditional02 distribution
            logb_0, loga_0 = self.curve4_conditional_logb_offsets
            r = np.random.uniform(*self.curve4_logb_range, n_channels).astype(np.float32)
            logb = logb_0 + r * (loga - loga_0)
            b = np.exp(logb)
        elif self.curve4_loga_weight is not None:
            # conditional01 distribution
            r = np.random.uniform(*self.curve4_logb_range, n_channels).astype(np.float32)
            logb = r + self.curve4_loga_weight * loga
            b = np.exp(logb)
        else:
            b = np.random.uniform(*self.curve4_b_range, n_channels).astype(np.float32)
        mirror_mask = np.random.randint(low=0, high=2, size=(3, 1)).astype(np.float32) if self.mirror_curve4 else None
        curves.append(Curve4(a, b, bp, bp2, self.bp_clip, wp, mirror_mask))

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
            if self.DEBUG:
                for channel, curve in enumerate(mask):
                    if curve[0]:
                        print(f'Curve0(gamma={gamma[channel]}, bp={bp[channel]}, bp2={bp2[channel]}, wp={wp[channel]})')
                        if self.param_csv_file is not None:
                            self.param_csv_file.write(f'curve0,{gamma[channel]},0,{bp[channel]},{bp2[channel]}\n')
                    if curve[1]:
                        print(f'Curve3(alpha={alpha[channel]}, beta={beta[channel]}, bp={bp[channel]}, bp2={bp2[channel]})')
                        if self.param_csv_file is not None:
                            self.param_csv_file.write(f'curve3,{alpha[channel]},{beta[channel]},{bp[channel]},{bp2[channel]}\n')
                    if curve[2]:
                        print(f'Curve4(a={a[channel]}, b={b[channel]}, bp={bp[channel]}, bp2={bp2[channel]})')
                        if self.param_csv_file is not None:
                            self.param_csv_file.write(f'curve4,{a[channel]},{b[channel]},{bp[channel]},{bp2[channel]}\n')
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
            # return both curves as "target"
            if self.predict_inverse:
                target = torch.cat((target, tfm), dim=1)

            # append rnd_factor for noise_level -> (C, 257)
            if self.noise_level:
                rnd_factor = torch.rand(1).repeat(3)
                target = torch.cat((target, rnd_factor[:, None], tfm), dim=1)

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



class CurveRMSE(MeanSquaredError):
    def __init__(self, curve='gamma', squared=False):
        super().__init__(squared=squared)
        self.curve = curve

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        if self.curve == 'gamma':
            support = torch.linspace(0, 1, 256, dtype=preds.dtype)
            # shapes: (1, 1, 256) ** (N, 3, 1) -> (N, 3, 256)
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
