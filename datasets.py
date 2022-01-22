import os
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import PIL.Image
# Alternatives to PIL (no speed gain):
# from skimage import io
import cv2

DEBUG = False


def show_image(image):
    """Show image"""
    plt.imshow(image)


class ImageDataset(Dataset):
    """Dataset for test images (no targets needed)"""

    def __init__(self, df, cfg, mode='train', transform=None, tensor_transform=None,
                 class_column='category_id'):
        """
        Args:
            df (pd.DataFrame): First row must contain the image file paths
            image_root (string, Path): Root directory for df.image_path
            transform (callable, optional): Optional transform to be applied
                on the first element (image) of a sample.
        """
        self.df = df.reset_index(drop=True)
        self.image_root = cfg.image_root
        self.transform = transform
        self.tensor_transform = tensor_transform
        self.labeled = (mode in ['train', 'valid'])
        self.multilabel = cfg.multilabel
        if self.labeled:
            n_classes = cfg.n_classes
            if self.multilabel:
                oh_columns = cfg.classes or self.df.columns[1:n_classes + 1]
                assert len(oh_columns) == n_classes, f'{len(oh_columns)} oh_columns but {n_classes} classes'
                self.labels = torch.FloatTensor(self.df[oh_columns].values)
            else:
                class_column = class_column or self.df.columns[1]
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
        if DEBUG: print(f"Image paths from column {self.df.columns[0]}")
        if self.labeled:
            if DEBUG: print(f"Labels from column {oh_columns if self.multilabel else class_column}")

        if cfg.xla and False:
            # Try GCS paths... breaks dataloader
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
            image = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)
            #image = PIL.Image.fromarray(image)  # optional, not required
            #image = torch.from_numpy(cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)).permute(2,0,1).contiguous()/255.
            #image = torch.FloatTensor(cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)).to(device) / 255
            #image = PIL.Image.open(fn)

        if self.transform:
            if self.transform.__module__.startswith('albumentations'):
                image = self.transform(image=np.array(image))['image']
                image = (image / 255).float()
            else:
                # torchvision, requires PIL.Image
                image = self.transform(PIL.Image.fromarray(image))

        if self.tensor_transform:
            image = self.tensor_transform(image)

        if self.labeled:
            label = self.labels[index]
            return (image, label)
        else:
            return (image,)


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
