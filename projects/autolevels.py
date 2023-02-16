import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

gcs_paths = {
    #'cassava-leaf-disease-classification': 'gs://kds-2f1ef51620b1d194f7b5370df991a8cf346870f5e8d86f95e5b81ba3',
    }

def init(cfg):
    cfg.competition_path = Path('/kaggle/input/imagenet-object-localization-challenge')
    if cfg.cloud == 'drive':
        cfg.competition_path = Path(f'/content/gdrive/MyDrive/{cfg.project}')

    if cfg.filetype == 'JPEG':
        cfg.image_root = cfg.competition_path / 'ILSVRC/Data/CLS-LOC/train'

    cfg.meta_csv = cfg.competition_path / 'ILSVRC/ImageSets/CLS-LOC/train_cls.txt'

    def read_csv(cfg):
        return pd.read_csv(cfg.meta_csv, sep=' ', usecols=[0], header=None, names=['image_id'])

    #elif cfg.filetype == 'tfrec':
    #    cfg.image_root = cfg.competition_path / 'train_tfrecords'

    #cfg.meta_csv = cfg.competition_path / 'train.csv'  # label
    #cfg.gcs_filter = 'train_tfrecords/*.tfrec'
    #if 'tf' in cfg.tags:
    #    cfg.n_classes = 5  # pytorch: set by metadata
    #cfg.gcs_paths = gcs_paths

    # Customize data pipeline (see tf_data for definition and defaults)
    #cfg.tfrec_format = {
    #    'image': tf.io.FixedLenFeature([], tf.string),
    #    'box': tf.io.FixedLenFeature([4], tf.int64),
    #    'target': tf.io.FixedLenFeature([], tf.int64)}
    #cfg.data_format = {
    #    'image': 'image',
    #    #'bbox': 'box',
    #    'target': 'target'}
    #cfg.inputs = ['image']
    #cfg.targets = ['target']

    cfg.dataset_class = ColorCastDataset


class Curve(torch.nn.Module):
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
        df['x'] = pdf(df.y.to_numpy(), alpha, beta, blackpoint)
        df.x = df.x.astype(int)  # pitfall: np.uint64 + int -> np.float64
        m = df.groupby('x').y.mean()  # sorted int index
        if len(m) < 1 + m.index[-1] - m.index[0]:
            print("using interpolation")
            m = m.reindex(range(m.index[0], m.index[-1] + 1)).interpolate('linear')
        assert len(m) == 1 + m.index.max() - m.index.min(), f"m.index: {m.index.min()} ... {m.index.max()} ({len(m)})"
        shape = x.shape
        x = np.clip(x, a_min=m.index[0], a_max=m.index[-1])
        x = x.astype(np.uint64).reshape(-1)
        return m.loc[x].to_numpy().reshape(shape)


class ColorCastDataset(Dataset):
    """Dataset for test images (no targets needed)"""

    def __init__(self, df, cfg, labeled=True, transform=None, tensor_transform=None,
                 class_column='category_id', return_path_attr=None):
        """
        Args:
            df (pd.DataFrame):                First row must contain the image file paths
            image_root (string, Path):        Root directory for df.image_path
            transform (callable, optional):   Optional transform to be applied on the first
                                              element (image) of a sample.
            labeled (bool, optional):         return `class_column` of `df`
            return_path_attr (str, optional): return Path attribute `return_path_attr`

        """
        self.df = df.reset_index(drop=True)
        self.image_root = cfg.image_root
        self.return_path_attr = return_path_attr
        self.labeled = labeled
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
        assert n_channels in {1, 3}, f'wrong image shape: {image.shape}, expecting channels last'
        
        labels = torch.stack([dist.sample((n_channels,)) for dist in [self.dist_a, self.dist_b, self.dist_bp]])
        assert labels.shape == (3, n_channels)

        for channel, label in enumerate(labels.T):
            curve = Curve(*label)
            image[:, :, channel] = curve.inverse_pdf(image[:, :, channel])

        if self.transform:
            transformed = self.transform(image=np.array(image), mask=mask)
            image = transformed["image"]
            image = (image / 255).float()
            mask = transformed["mask"]  # -> uint8 tensor

        if self.tensor_transform:
            image = self.tensor_transform(image)

        return image, labels if self.labeled else image