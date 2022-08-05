from pathlib import Path
from multiprocessing import cpu_count

import torch
import webdataset as wds
from augmentation import get_tfms, get_torchvision_tfms
from tf_data import count_data_items
from webdataset.utils import identity


def patch_len_call(instance, func):
    class _(type(instance)):
        def __len__(self, *arg, **kwarg):
            return func(*arg, **kwarg)
    instance.__class__ = _


def patch_len_value(instance, value):
    class _(type(instance)):
        def __len__(self, *arg, **kwarg):
            return value
    instance.__class__ = _


def split_train_files(cfg, use_fold):
    """Split cfg.train_files into train_shards/valid_shards"""
    shards = cfg.train_files

    # split by name
    train_shards = [s for s in shards if '-train-' in Path(s).name]
    valid_shards = sorted(set(shards) - set(train_shards))
    if train_shards and valid_shards:
        return train_shards, valid_shards

    # split by use_fold, num_folds
    train_shards = [s for i, s in enumerate(shards) if i % cfg.num_folds != use_fold]
    valid_shards = [s for i, s in enumerate(shards) if i % cfg.num_folds == use_fold]
    return train_shards, valid_shards


def get_dataloader(cfg, use_fold, xm, mode='train', **kwargs):

    if mode == 'test':
        shards = cfg.test_files
    else:
        train_shards, valid_shards = split_train_files(cfg, use_fold)
        shards = train_shards if mode == 'train' else valid_shards

    shuffle = 2048 if mode == 'train' else None
    drop_last = (mode == 'train')
    resampled = (cfg.n_replicas > 1) and (mode == 'train')

    # Calculate iterations / epoch ()
    n_examples = count_data_items(shards)  # requires shard names "*-{n_examples}.*"
    n_replica = cfg.n_replicas
    n_workers = kwargs.get('num_workers', 0) or 1
    n_dataset_instances = n_replica * n_workers        # n_workers = 1 if xla anyways
    ds_epoch_size = n_examples // n_dataset_instances  # n_replica? not sure about this...
    if (mode != 'train') and (n_examples % n_dataset_instances > 0):
        ds_epoch_size += 1  # also do final partial batch

    if cfg.frac < 1:
        xm.master_print(f"Sampling {int(n_examples * cfg.frac)} out of {n_examples} examples")

    steps_per_epoch = int(n_examples * cfg.frac) // (cfg.bs * n_replica * cfg.n_acc)
    if drop_last:
        mini_batches_per_epoch = steps_per_epoch * n_replica * cfg.n_acc
        macro_batches_per_epoch = steps_per_epoch * cfg.n_acc
    else:
        mini_batches_per_epoch = int(n_examples * cfg.frac) // cfg.bs
        if n_examples % cfg.bs > 0: mini_batches_per_epoch += 1
        macro_batches_per_epoch = mini_batches_per_epoch // n_replica
        if mini_batches_per_epoch % n_replica > 0: macro_batches_per_epoch += 1

    tfms = get_tfms(cfg, mode=mode) if cfg.use_albumentations else get_torchvision_tfms(cfg, mode=mode)
    # albu wants named args: aug(image=image)
    # torchvision wants pil.image not array
    if cfg.DEBUG:
        xm.master_print(f'{mode} tfms:')
        xm.master_print(tfms)

    def map_albu(image):
        image = tfms(image=image)['image']
        return image.float() / 255.0

    def tensorize(inputs):
        #assert len(inputs) == 2, f'expected 2 but got {len(inputs)} inputs'
        img, label = inputs
        #print("img:", type(img), img.shape, img.dtype)  # unbatched uint8 array
        #print("label:", type(label))  # int
        #print("image writable?", img.flags['WRITEABLE'])
        img = torch.FloatTensor(img.copy()).permute((2, 0, 1)).contiguous() / 255.0  # w/o copy(): not-writable warning
        label = torch.tensor(label, dtype=torch.int64)
        return img, label

    dataset = wds.WebDataset(shards, shardshuffle=bool(shuffle), resampled=resampled)
    dataset = dataset.shuffle(shuffle) if shuffle else dataset
    if cfg.use_albumentations:
        dataset = dataset.decode("rgb8").to_tuple("jpeg", "cls").map_tuple(map_albu, identity)  # albumentations
    else:
        dataset = dataset.decode("pil").to_tuple("jpeg", "cls").map_tuple(tfms, identity)  # torchvision
        #dataset = dataset.decode("rgb8").to_tuple("jpeg", "cls").map(tensorize)
    dataset = dataset.batched(cfg.bs, partial=drop_last) if not cfg.use_batch_tfms else dataset
    dataset = dataset.with_epoch(ds_epoch_size) if resampled else dataset

    #loader = torch.utils.data.DataLoader(dataset, batch_size=None, **kwargs)
    loader = wds.WebLoader(dataset, batch_size=None, **kwargs)  # same but adds "fluid interface"

    # Suggested as "most efficient" but host runs OOR (and first iters are even slower):
    #loader = loader.unbatched().shuffle(shuffle//4).batched(cfg.bs, partial=drop_last) if mode == 'train' else loader

    # ParallelLoader calls __len__ on loader, which raises ValueError (has no len).
    # Pytorch's DataLoader.__len__ provides and estimate for len(dataset) / per_replica_bs
    # Schedulers raise ValueError if stepped more than total_steps times.
    # loader.with_epoch() sets the number of batches its iterator yields (nsamples).
    # loader.with_length() just sets the value returned by len(), no change to its iterator.
    # => set loader.nsamples so that it applies "cfg.frac"
    # => set loader.__len__ so that ParallelLoader behaves as expected.
    # MpDeviceLoader does not divide len by n_replicas (DistributedSampler presumably does that)
    # => use .with_length(macro_batches_per_epoch)
    loader = loader.with_epoch(macro_batches_per_epoch).with_length(mini_batches_per_epoch)
    xm.master_print(f"\n{mode} DataLoader")
    xm.master_print("Batches / epoch:   ", mini_batches_per_epoch)   # OK
    xm.master_print("Iterations / epoch:", macro_batches_per_epoch)  # OK
    xm.master_print("Steps / epoch:     ", steps_per_epoch)          # OK
    xm.master_print("ds_epoch_size:     ", ds_epoch_size)            # no sense
    xm.master_print("len(dataloader):   ", len(loader))              # OK
    xm.master_print("dataset.nsamples:  ", dataset.nsamples)         # no sense (=ds_epoch_size)
    xm.master_print("loader.nsamples:   ", loader.nsamples)          # OK (=iter/epoch)
    xm.master_print("label.shape from loader:", next(iter(loader))[1].shape)  # OK, replica_bs

    return loader


def get_dataloaders(cfg, use_fold, xm):

    train_loader = get_dataloader(cfg, use_fold, xm, mode='train',
                                  num_workers=1 if cfg.n_replicas > 1 else cpu_count(),
                                  pin_memory=True)

    valid_loader = get_dataloader(cfg, use_fold, xm, mode='valid',
                                  num_workers=1 if cfg.n_replicas > 1 else cpu_count(),
                                  pin_memory=True)

    xm.master_print("train images:", cfg.NUM_TRAINING_IMAGES)
    xm.master_print("valid images:", cfg.NUM_VALIDATION_IMAGES)
    xm.master_print("test images: ", cfg.NUM_TEST_IMAGES)

    if cfg.xla and (cfg.deviceloader == 'mp'):
        from torch_xla.distributed.parallel_loader import MpDeviceLoader
        device = xm.xla_device()
        train_loader = MpDeviceLoader(train_loader.with_length(len(train_loader) // cfg.n_replicas), device)
        valid_loader = MpDeviceLoader(valid_loader.with_length(len(train_loader) // cfg.n_replicas), device)
        # When iterated in train_fn: bs on replica is correct (cfg.bs) and each rank gets a different batch
        # => len(iterable) should be divided by cfg.n_replicas
        xm.master_print(f"len(train_loader):", len(train_loader))
        xm.master_print(f"len(valid_loader):", len(valid_loader))

    return train_loader, valid_loader
