import os
from pathlib import Path
from multiprocessing import cpu_count
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, utils
import PIL.Image
# Alternatives to PIL (no speed gain):
# from skimage import io
import cv2

from augmentation import get_tfms

DEBUG = False


def show_image(image):
    """Show image"""
    plt.imshow(image)


class ImageDataset(Dataset):
    """Dataset for test images (no targets needed)"""

    def __init__(self, df, cfg, labeled=True, transform=None, tensor_transform=None,
                 return_path_attr=None):
        """
        Args:
            df (pd.DataFrame):                First row must contain the image file paths
            image_root (string, Path):        Root directory for df.image_path
            transform (callable, optional):   Optional transform to be applied on the first
                                              element (image) of a sample.
            labeled (bool, optional):         return `cfg.class_column` of `df`
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
        self.multilabel = cfg.multilabel

        if self.labeled:
            n_classes = cfg.n_classes
            if self.multilabel:
                oh_columns = cfg.classes or self.df.columns[1:n_classes + 1]
                assert len(oh_columns) == n_classes, f'{len(oh_columns)} oh_columns but {n_classes} classes'
                self.labels = torch.FloatTensor(self.df[oh_columns].values)
            else:
                class_column = cfg.class_column or self.df.columns[1]
                self.labels = torch.LongTensor(self.df[class_column].values)
                if len(self) > 0:
                    min_label, max_label = self.labels.min(), self.labels.max()
                    assert min_label >= 0
                    if n_classes == 1:
                        self.labels = torch.FloatTensor(self.df[class_column].values[:, None])
                        print("n_classes = 1: use BCE loss!")
                    elif n_classes:
                        assert max_label < n_classes, f"largest label {max_label} >= n_classes {n_classes}"
                    else:
                        n_classes = max_label + 1
            if DEBUG: print(f"Labels from column {oh_columns if self.multilabel else class_column}")
        if DEBUG: print(f"Image paths from column {self.df.columns[0]}")

        if cfg.xla and False:
            # Try GCS paths... fixme!
            import io
            from torch_xla.utils import gcsfs

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        #if torch.is_tensor(index):
        #    index = index.tolist()

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

        if self.tensor_transform:
            image = self.tensor_transform(image)

        labels = []
        if self.labeled:
            labels.append(self.labels[index])
        if self.return_path_attr:
            labels.append(getattr(Path(fn), self.return_path_attr))  # colates to tuple of str

        if len(labels) > 1:
            return image, tuple(labels)
        elif len(labels) == 1:
            return (image, labels[0])
        else:
            return image


class MySiimCovidAuxDataset(Dataset):
    def __init__(self, df, cfg, mode='train', transform=None, tensor_transform=None):
        self.df = df.reset_index(drop=True)
        self.image_root = cfg.image_root
        self.transform = transform
        self.tensor_transform = tensor_transform
        self.oh_columns = cfg.classes or self.df.columns[1:cfg.n_classes + 1]
        self.labeled = (mode in ['train', 'valid'])
        assert cfg.multilabel and cfg.use_aux_loss

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fn = os.path.join(self.image_root, self.df.iloc[index, 0])
        image = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)

        cls_label = torch.FloatTensor(self.df.iloc[index].loc[self.oh_columns])

        height, width = image.shape[0:2]
        orig_height, orig_width = self.df.iloc[index].loc[['height', 'width']]

        mask = np.zeros((height, width), dtype=np.uint8)
        if self.df.label.iloc[index].startswith('opacity'):
            arr = self.df.label.iloc[index].split(' ')
            nums = len(arr) // 6
            assert nums > 0

            for i in range(nums):
                class_name = arr[6 * i]
                assert class_name == 'opacity'
                x1 = int(float(arr[6 * i + 2]) * width / orig_width)
                y1 = int(float(arr[6 * i + 3]) * height / orig_height)
                x2 = int(float(arr[6 * i + 4]) * width / orig_width)
                y2 = int(float(arr[6 * i + 5]) * height / orig_height)
                #print(f"xyxy: {x1, y1, x2, y2}       image: {width, height}")
                x1 = min(max(0, x1), width)
                x2 = min(max(0, x2), width)
                y1 = min(max(0, y1), height)
                y2 = min(max(0, y2), height)
                if x1 >= x2 or y1 >= y2: continue
                mask[y1:y2, x1:x2] = np.ones((y2 - y1, x2 - x1), dtype=np.uint8)
            #print(f"mask with {nums} boxes: {mask.dtype, mask.min(), mask.max()}")

        if self.transform:
            transformed = self.transform(image=np.array(image), mask=mask)
            image = transformed["image"]
            image = (image / 255).float()
            mask = transformed["mask"]  # -> uint8 tensor
        if self.tensor_transform:
            image = self.tensor_transform(image)
            mask = transforms.ToTensor()(mask)
        mask = mask.float()
        mask = torch.unsqueeze(mask, 0)

        if self.labeled:
            return image, mask, cls_label
        else:
            return image


class ImageDataLoader(DataLoader):
    """Dataloader for test images"""

    def show_batch(self, max_n: int = 16, n_cols: int = 4, padding: int = 2,
                   normalize: bool = False, pad_value: int = 0,
                   figsize: Tuple[int, int] = (12, 6)):
        """Show images in a batch of samples."""
        batch = next(iter(self))
        images_batch = batch[0][:max_n]
        plt.figure(figsize=figsize)
        grid = utils.make_grid(images_batch, n_cols, padding, normalize,
                               pad_value=pad_value)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.title('Batch from dataloader')


def get_fakedata_loaders(cfg, device):
    from torch_xla.utils.utils import SampleGenerator

    train_loader = SampleGenerator(
        data=(torch.zeros(cfg.bs, 3, *cfg.size),
              torch.zeros(cfg.bs, dtype=torch.int64)),
        sample_count=cfg.NUM_TRAINING_IMAGES // (cfg.bs * cfg.n_replicas))

    valid_loader = SampleGenerator(
        data=(torch.zeros(cfg.bs, 3, *cfg.size),
              torch.zeros(cfg.bs, dtype=torch.int64)),
        sample_count=cfg.NUM_VALIDATION_IMAGES // (cfg.bs * cfg.n_replicas))

    if cfg.xla and (cfg.deviceloader == 'mp'):
        from torch_xla.distributed.parallel_loader import MpDeviceLoader

        train_loader = MpDeviceLoader(train_loader, device)
        valid_loader = MpDeviceLoader(valid_loader, device)

    return train_loader, valid_loader


def get_dataloaders(cfg, use_fold, metadata, xm, augment=True):

    if cfg.filetype == 'wds':
        from experimental import web_datasets
        return web_datasets.get_dataloaders(cfg, use_fold, xm)  # xm: only for printing

    elif cfg.filetype == 'tfds':
        from experimental import tfds
        return tfds.get_dataloaders(cfg, use_fold)

    is_valid = metadata.is_valid if hasattr(metadata, 'is_valid') else (metadata.fold == use_fold)
    is_shared = (metadata.fold == cfg.shared_fold) if cfg.shared_fold is not None else False
    meta_train = metadata.loc[~ is_valid]
    meta_valid = metadata.loc[is_valid | is_shared]

    # Create datasets
    train_tfms = get_tfms(cfg, mode='train')
    test_tfms = get_tfms(cfg, mode='test')
    tensor_tfms = None

    dataset_class = cfg.dataset_class or ImageDataset

    ds_train = dataset_class(meta_train, cfg, labeled=True,
                             transform=train_tfms if augment else test_tfms, tensor_transform=tensor_tfms)
    
    ds_valid = dataset_class(meta_valid, cfg, labeled=True,
                             transform=test_tfms, tensor_transform=tensor_tfms)

    xm.master_print("ds_train:", len(ds_train))
    xm.master_print("ds_valid:", len(ds_valid))
    if cfg.DEBUG:
        xm.master_print("train_tfms:")
        xm.master_print(ds_train.transform)
        xm.master_print("test_tfms:")
        xm.master_print(ds_valid.transform)

    # Create samplers for distributed shuffling or class-balancing
    if cfg.xla:
        from catalyst.data import DistributedSamplerWrapper
        from torch.utils.data.distributed import DistributedSampler

    if cfg.do_class_sampling:
        # Data samplers, class-weighted metrics
        train_labels = meta_train[cfg.class_column]

        # Use torch's WeightedRandomSampler with custom class weights
        class_counts = train_labels.value_counts().sort_index().values
        n_examples = len(train_labels)
        assert class_counts.sum() == n_examples, f"{class_counts.sum()} != {n_examples}"
        class_weights_full = n_examples / (cfg.n_classes * class_counts)
        class_weights_sqrt = np.sqrt(class_weights_full)
        sample_weights = class_weights_sqrt[train_labels.values]
        assert len(sample_weights) == n_examples
        #xm.master_print("Class counts:           ", class_counts)
        #xm.master_print("Class weights:          ", class_weights_full)
        #xm.master_print("Weighted class counts:  ", class_weights_full * class_counts)
        xm.master_print(f"Sampling {int(n_examples * cfg.frac)} out of {n_examples} examples")

        train_sampler = WeightedRandomSampler(
            sample_weights, int(n_examples * cfg.frac), replacement=True)

        train_sampler = DistributedSamplerWrapper(
            train_sampler,
            num_replicas = cfg.n_replicas,
            rank         = xm.get_ordinal(),
            shuffle      = False) if (cfg.n_replicas > 1) else train_sampler

    elif cfg.n_replicas > 1:
        train_sampler = DistributedSampler(
            ds_train,
            num_replicas = cfg.n_replicas,
            rank         = xm.get_ordinal(),
            shuffle      = True)

    else:
        train_sampler = None

    valid_sampler = DistributedSampler(ds_valid,
                                       num_replicas = cfg.n_replicas,
                                       rank         = xm.get_ordinal(),
                                       shuffle      = False,
                                       ) if cfg.n_replicas > 1 else None

    # Create dataloaders
    train_loader = DataLoader(ds_train,
                              batch_size  = cfg.bs,
                              sampler     = train_sampler,
                              num_workers = 1 if cfg.n_replicas > 1 else cpu_count(),
                              pin_memory  = True if cfg.gpu else False,
                              drop_last   = True,
                              shuffle     = False if train_sampler else True)

    valid_loader = DataLoader(ds_valid,
                              batch_size  = cfg.bs,
                              sampler     = valid_sampler,
                              num_workers = 1 if cfg.n_replicas > 1 else cpu_count(),
                              pin_memory  = True if cfg.gpu else False)

    if cfg.xla and (cfg.deviceloader == 'mp'):
        from torch_xla.distributed.parallel_loader import MpDeviceLoader
        device = xm.xla_device()
        train_loader = MpDeviceLoader(train_loader, device)
        valid_loader = MpDeviceLoader(valid_loader, device)

    return train_loader, valid_loader


def get_test_loader(cfg, metadata, xm, return_path_attr=None):

    test_tfms = get_tfms(cfg, mode='test')

    ds_test = ImageDataset(metadata, cfg, labeled=False, return_path_attr='stem',
                           transform=test_tfms, tensor_transform=None)

    xm.master_print("ds_test:", len(ds_test))
    if cfg.DEBUG:
        xm.master_print("test_tfms:")
        xm.master_print(ds_test.transform)

    # Create samplers for distributed shuffling or class-balancing
    if cfg.xla:
        from torch.utils.data.distributed import DistributedSampler

    test_sampler = DistributedSampler(ds_test,
                                      num_replicas = cfg.n_replicas,
                                      rank         = xm.get_ordinal(),
                                      shuffle      = False,
                                      ) if cfg.n_replicas > 1 else None

    # Create dataloader
    test_loader = DataLoader(ds_test,
                             batch_size  = cfg.bs,
                             sampler     = test_sampler,
                             num_workers = 1 if cfg.n_replicas > 1 else cpu_count(),
                             pin_memory  = True if cfg.gpu else False)

    if cfg.xla and (cfg.deviceloader == 'mp'):
        from torch_xla.distributed.parallel_loader import MpDeviceLoader
        device = xm.xla_device()
        test_loader = MpDeviceLoader(test_loader, device)

    return test_loader
