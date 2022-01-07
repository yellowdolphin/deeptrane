import os
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 100)

DEBUG = False


### Metadata, Labels ----------------------------------------------------------

def get_metadata(cfg):
    "Return a DataFrame with labels, image paths, dims, splits required by the datasets."
    
    if 'pretrain' in cfg.tags:
        competition_path, meta_csv = Path('/kaggle/input/data'), 'Data_Entry_2017.csv'
        df = pd.read_csv(competition_path/meta_csv)
        df['image_id'] = df['Image Index'].str.split('.').str[0]
    else:
        competition_path, meta_csv = Path('/kaggle/input/siim-covid19-detection'), 'train_image_level.csv'
        df = pd.read_csv(competition_path/meta_csv)
        df['image_id'] = df.id.str.split('_').str[0]
    if DEBUG: print(df.head(3))

    if 'image' in cfg.tags or ('add_5th_class' in cfg and cfg.add_5th_class):
        df = add_2class_label(df)

    if 'study' in cfg.tags:
        df = add_study_label(df, cfg)  # study labels stored in oh_columns and/or 'category_id'
    elif 'pretrain' in cfg.tags:
        df = add_chest14_labels(df, cfg)

    df = add_filename(df, cfg)

    df = maybe_encode_labels(df, cfg)

    required_columns = ['image_id', 'image_path', 'category_id']
    if cfg.multilabel: 
        required_columns.extend(cfg.classes)
        # keep 'category_id' for StratifiedKFold
    if cfg.use_aux_loss:
        required_columns.append('label')        
    df = df[required_columns].reset_index(drop=True)
    if DEBUG: print(df.head(3))

    if cfg.use_aux_loss:
        # To properly scale segmentation masks, add original image dims
        meta_csv = os.path.join(cfg.image_root.parent, 'meta.csv')
        df = add_image_dims(df, meta_csv)

    df = split_data(df, cfg)

    return df

def get_class_ids(s):
    "Return list of class_ids from prediction_string `s`"
    a = np.array(s.split()).reshape(-1, 6)
    return set(a[:, 0].tolist())

def get_single_label_class_ids(s):
    s = get_class_ids(s)
    assert len(s) == 1
    return s.pop()

def add_2class_label(metadata, class_column='category_id'):
    # Image labels are all singlelabels!
    metadata[class_column] = metadata.label.apply(get_single_label_class_ids)
    print(f"{len(metadata)} examples")
    if 'image_id'         in metadata: print(f"{metadata.image_id.nunique()} unique image ids")
    if 'StudyInstanceUID' in metadata: print(f"{metadata.StudyInstanceUID.nunique()} unique study ids")
    if DEBUG: print(metadata.head(3))
    return metadata

def add_study_label(metadata, cfg, singlelabel_column='category_id'):
    competition_path = Path('/kaggle/input/siim-covid19-detection') 
    meta_csv = 'train_study_level.csv'
    train_labels_study = pd.read_csv(competition_path/meta_csv)
    train_labels_study['StudyInstanceUID'] = train_labels_study.id.str.split('_').str[0]

    oh_columns = ['Negative for Pneumonia', 'Typical Appearance',
                  'Indeterminate Appearance', 'Atypical Appearance']
    train_labels_study['study_label'] = train_labels_study[oh_columns].to_numpy().argmax(axis=1)

    # Add study_label to images associated with study.
    metadata = metadata.merge(train_labels_study[['StudyInstanceUID', 'study_label'] + oh_columns],
                              on='StudyInstanceUID').drop(columns='boxes')
    if not cfg.use_aux_loss:
        metadata.drop(columns='label', inplace=True)

    # 5-class multilabel classifier: study-labels + image-level "opacity" as 5th class
    if 'add_5th_class' in cfg and cfg.add_5th_class:
        assert cfg.multilabel, '5th study class makes only sense with multilabel=True'
        metadata['Opacity'] = (metadata[singlelabel_column] == 'opacity').astype(int)
        oh_columns.append('Opacity')

    if not cfg.multilabel:
        metadata.drop(columns=oh_columns, inplace=True)

    metadata[singlelabel_column] = metadata.study_label
    metadata.drop(columns='study_label', inplace=True)
    
    # Filter images to be included in multi-image studies
    if 'exclude_multiimage_studies' in cfg and cfg.exclude_multiimage_studies:
        # Round 1: Only single-image studies
        images_per_study = metadata.groupby('StudyInstanceUID').image_id.count()
        single_image_studies = images_per_study[images_per_study == 1].index
        metadata = metadata.set_index('StudyInstanceUID').loc[single_image_studies]
    else:
        # Filter out badly labeled image_ids
        # Unlabelled duplicates from https://www.kaggle.com/kwk100/siim-covid-19-duplicate-training-images
        bad_image_ids = set('''
        00c1515729a8 2c130ee08736 684230477525 be65d1a22de5 c843d08b49b8 6534a837497d ea2117b53323 eea3a910fa9e 
        1ea01196514a 9b1de1c45491 3b6ad60071d4 04cc2f7f4c4b 05c063f5cef5 156cb1f5c689 4c414b793562 a5a364383f34 b121806162c3 
        bee62c601ae9 c6e92e59a0ae 72cf260ddf4c 00c1515729a8 6ff218d10741 7f1924880cf8 32222cc776a2 99f3642f50f5 12e97ed89297 
        ab55abb953d1 30b18db28900 550f057ee0b0 b5a415f70aa9 2c130ee08736 684230477525 39d52f244db3 611348a721f7 deea6d6f81a5 
        2f973757eaa4 1b92142f4362 4cbc17936e7d 582c442e440b 6e9fad584bff 0102b5cac730 b3ffe59e37c7 7626c521cad7 08acae0bf785 
        35e398a5a431 6f54e9cbd180 c3a09e8a600d c4b68b29a072 9939f63af4ff e3e2f20e0264 be65d1a22de5 c843d08b49b8 a57bf6dd6119 
        9fcbe25a88e0 717ea5155b46 2a7a456d367e 8d4b3609ed92 9872a8a48f23 e738c549fe8e 149356f04849 6d36ffbc7864 ce51b397b1a6 
        608d574388ba 6728e11290af 866e3622cb24 df4f1240317e 267a250932bc 49664f078f0e 869476b0763a a39667fe9a81 b97c6b32105e 
        ddb051c1233b ef6e312ca719 8093df07a5d0 779f0040d1b2 76c66ee8e58d 66dabc6f972d df2bb22fa871 6e5946091b8a 75b52bec817f 
        3577ee4f26c4 a94171e98807 a5bbd30ed109'''.split())
        mask = metadata.image_id.isin(bad_image_ids)
        if DEBUG: print(f"Filtering out {mask.sum()} unlabelled images")
        metadata = metadata.loc[~mask].copy()
        # In multi-image studies, take only one example (rest are duplicates)
        metadata.drop_duplicates(subset='StudyInstanceUID', keep='last', inplace=True)
        #metadata.drop_duplicates(subset=['StudyInstanceUID'] + oh_columns[:4], keep='last', inplace=True) # same result

    cfg['classes'] = oh_columns

    if DEBUG: 
        print(f"{len(metadata)} examples")
        print(metadata.head(3))

    return metadata

def add_chest14_labels(metadata, cfg, singlelabel_column='category_id'):
    labels = metadata['Finding Labels'].str.split('|').apply(set)
    # Skip some irrelevant classes
    #labels = labels.apply(lambda x: x.difference(['No Finding', 'Cardiomegaly']))
    #labels = labels.apply(lambda x: x.difference(['Cardiomegaly']))
    if cfg.multilabel:
        oh_columns = ['Cardiomegaly', 'Emphysema', 'Effusion', 'No Finding', 'Hernia', 'Infiltration',
                      'Mass', 'Nodule', 'Atelectasis', 'Pneumothorax', 'Pleural_Thickening',
                      'Pneumonia', 'Fibrosis', 'Edema', 'Consolidation']
        for c in oh_columns:
            metadata[c] = labels.apply(set([c]).issubset).astype(int)
        # Add singlelabel column for StratifiedKFold splitting
        metadata[singlelabel_column] = metadata[oh_columns].to_numpy().argmax(axis=1)
        assert 'classes' not in cfg, 'cannot overwrite class names in cfg'
        cfg.classes = oh_columns
    else:
        # Extract single-label instances from multilabel Chest14 dataset
        singlelabel = labels.apply(len) < 2
        labels = labels.apply(lambda x: 'none' if len(x) == 0 else x.pop())
        metadata[singlelabel_column] = labels
        metadata = metadata[singlelabel]

    if DEBUG: print(metadata.head(5))
    return metadata

def add_filename(metadata, cfg):
    n_parts = 1
    if 'pretrain' in cfg.tags:
        datasets = [f'/kaggle/input/data/images_{i:03d}/images' for i in range(1,13)]
        datatype = 'png'
    elif cfg.size[0] <= 224:
        datasets = ['/kaggle/input/siim-covid19-resized-to-224px-png/train']
        datatype = 'png'
    elif cfg.size[0] <= 512:
        datasets = ['/kaggle/input/siim-covid19-resized-to-512px-png/train']
        datatype = 'png'
    else:
        datasets = ['/kaggle/input/siim-covid19-resized-to-1024px-jpg/train']
        datatype = 'jpg'

    # Copy data (colab) and set image_root path
    path = Path(datasets[0]) if os.path.exists('/kaggle') else Path('.')
    assert path.exists(), f"no folder {path}"
    if cfg.xla and False:  # dataloader.show_batch() hangs
        from kaggle_datasets import KaggleDatasets
        gcs_path = KaggleDatasets().get_gcs_path(path.parent.name)
        path = f'{gcs_path}/train'
    cfg.image_root = path
    print("[ √ ] image_root path:", cfg.image_root)

    # Add image_path above image_root to metadata
    if 'pretrain' in cfg.tags:
        file_names = {}
        for dataset in datasets:
            for fn in os.scandir(dataset):
                image_id = fn.name[:-4]
                file_names[image_id] = os.path.join(dataset, fn)
        metadata['image_path'] = metadata.image_id.apply(lambda i: file_names[i])    
    else:
        metadata['image_path'] = metadata.image_id + f'.{datatype}'

    return metadata

def maybe_encode_labels(metadata, cfg, class_column='category_id'):
    if cfg.multilabel:
        assert 'classes' in cfg, 'no multilabel class names found in cfg'
        cfg.n_classes = len(cfg.classes)
    else:
        cfg.n_classes = metadata[class_column].nunique()
        max_label, min_label = metadata[class_column].max(), metadata[class_column].min()
        if (metadata[class_column].dtype == 'O') or (max_label + 1 > cfg.n_classes) or (min_label < 0):
            from sklearn.preprocessing import LabelEncoder
            vocab = LabelEncoder()
            vocab.fit(metadata[class_column].values)
            metadata[class_column] = vocab.transform(metadata[class_column].values)
            import pickle
            pickle.dump(vocab, open(f'{cfg.out_dir}/vocab.pkl', 'wb'))
            print(f"[ √ ] Label encoder saved to '{cfg.out_dir}/vocab.pkl'")
            #print("backtrafo: vocab.inverse_transform(x) or vocab.classes_[x]")
            #print("classes:", vocab.classes_)
        else:
            print("[ √ ] No label encoding required.")
    print("[ √ ] n_classes:", cfg.n_classes)
    return metadata

def add_image_dims(metadata, meta_csv):
    dims = pd.read_csv(meta_csv).rename(columns={'dim0': 'height', 'dim1': 'width'})
    dims.drop(columns='split', inplace=True)
    metadata = metadata.merge(dims, on='image_id')
    if DEBUG: print(metadata.head(3))
    return metadata

def split_data(metadata, cfg, class_column='category_id', seed=42):
    if cfg.train_on_all:
        metadata['fold'] = 0
        train_idx, valid_idx = metadata.index, np.array([], dtype='int')
    else:
        if 'folds' in cfg and cfg.folds:
            # Get splitting from existing folds.json
            assert os.path.exists(cfg.folds), f'no file {cfg.folds}'
            folds = pd.read_json(cfg.folds)['image_id', 'fold']
            metadata = metadata.merge(folds, on='image_id')
        else:
            from sklearn.model_selection import StratifiedKFold

            labels = metadata[class_column].values
            skf = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=seed)

            metadata['fold'] = -1
            for fold, subsets in enumerate(skf.split(metadata.index, labels)):
                metadata.loc[subsets[1], 'fold'] = fold

    metadata.set_index('image_id', inplace=True)
    if DEBUG: print(metadata.head(5))
    return metadata


### Datasets, Dataloaders -----------------------------------------------------

from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import PIL
# Alternatives to PIL (no speed gain):
import cv2
#from skimage import io

def show_image(image):
    """Show image"""
    plt.imshow(image)

class ImageDataset(Dataset):
    """Dataset for test images (no targets needed)"""

    def __init__(self, df, cfg, mode='train', transform=None, tensor_transform=None, class_column='category_id'):
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
                oh_columns = cfg.classes or self.df.columns[1:n_classes+1]
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
            image = PIL.Image.fromarray(image)
            #image = torch.from_numpy(cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)).permute(2,0,1).contiguous()/255.
            #image = torch.FloatTensor(cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2RGB)).to(device) / 255
            #image = PIL.Image.open(fn)    # PIL.Image
        
        if self.transform:
            if self.transform.__module__.startswith('albumentations'):
                image = self.transform(image=np.array(image))['image']
                image = (image / 255).float()
            else:
                image = self.transform(image)
        
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
        self.oh_columns = cfg.classes or self.df.columns[1:cfg.n_classes+1]
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
                class_name = arr[6*i]
                assert class_name == 'opacity'
                x1 = int(float(arr[6*i+2]) * width / orig_width)
                y1 = int(float(arr[6*i+3]) * height / orig_height)
                x2 = int(float(arr[6*i+4]) * width / orig_width)
                y2 = int(float(arr[6*i+5]) * height / orig_height)
                #print(f"xyxy: {x1, y1, x2, y2}       image: {width, height}")
                x1 = min(max(0, x1), width)
                x2 = min(max(0, x2), width)
                y1 = min(max(0, y1), height)
                y2 = min(max(0, y2), height)
                if x1 >= x2 or y1 >= y2: continue
                mask[y1:y2, x1:x2] = np.ones((y2 - y1, x2 - x1), dtype=np.uint8)
            #print(f"mask with {nums} boxes: {mask.dtype, mask.min(), mask.max()}")

            if self.transform and use_albumentations:
                transformed = self.transform(image=np.array(image), mask=mask)
                image = transformed["image"]
                image = (image / 255).float()
                mask = transformed["mask"]
                #print(f"transformed mask:", mask.dtype, mask.min(), mask.max())
            elif self.transform:
                raise NotImplementedError("implement torchvision mask transform first")
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
    
    def show_batch(self, max_n: int=16, n_cols: int=4, padding: int=2, normalize: bool=False, 
                   pad_value: int=0, figsize: Tuple[int, int]=(12,6)):
        """Show images in a batch of samples."""
        batch = next(iter(self))
        images_batch = batch[0][:max_n]
        #batch_size = len(images_batch)
        #im_height = images_batch.size(2)
        plt.figure(figsize=figsize)
        grid = utils.make_grid(images_batch, n_cols, padding, normalize,
                               pad_value=pad_value)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.title('Batch from dataloader')