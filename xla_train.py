import PIL
from multiprocessing import cpu_count
from functools import partial
from pathlib import Path
from future import removesuffix
import time
import math
import random

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score
from metrics import val_map, multiclass_average_precision_score
from datasets import get_metadata, ImageDataset, MySiimCovidAuxDataset
import torch
torch.set_default_tensor_type('torch.FloatTensor')
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms.functional as TF
from torch.optim import lr_scheduler
import torch.optim as optim
from augmentation import get_tfms
from utils import listify
from models import is_bn

class AverageMeter(object):
    '''Computes and stores the average and current value'''
    
    def __init__(self, xm):
        self.xm = xm   # allow overload at runtime
        self.reset()

    def reset(self):
        self.val   = 0
        self.sum   = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
    
    @property
    def average(self):
        reduced_sum = self.xm.mesh_reduce('meter_sum', self.sum, sum)
        reduced_count = self.xm.mesh_reduce('meter_count', self.count, sum)
        return reduced_sum / reduced_count
    
    @property
    def current(self):
        # current value, averaged over devices (and minibatch)
        return self.xm.mesh_reduce('meter_val', self.val, reduce)


def reduce(values):
    if isinstance(values, torch.Tensor):
        return torch.mean(values)
    return sum(values) / len(values)

def get_one_cycle_scheduler(optimizer, dataset, max_lr, cfg, xm, rst_epoch=1, dataloader=None):
    # ToDo: Rid dataloader if its attributes prove useless
    if hasattr(dataloader, 'drop_last'): assert dataloader.drop_last is True
    frac = cfg.frac if cfg.do_class_sampling else 1
    n_acc = cfg.n_acc
    num_cores = cfg.num_tpu_cores if cfg.xla else 1
    steps_per_epoch = int(len(dataset) * frac) // (cfg.bs * num_cores * n_acc)
    total_steps = (rst_epoch + cfg.epochs) * steps_per_epoch
    last_step = rst_epoch * steps_per_epoch - 1 if rst_epoch else -1
    #xm.master_print(f"cfg.epochs: {cfg.epochs}, frac: {frac}, len(ds): {len(dataset)}, num_cores: {num_cores}")
    xm.master_print(f"Total steps: {total_steps} (my guess)")
    if rst_epoch != 0:
        xm.master_print(f"Restart epoch: {rst_epoch}")
    if hasattr(dataloader, '__len__'):
        xm.master_print(f"Total steps: {(rst_epoch + cfg.epochs) * (len(dataloader) // n_acc)} (dataloader)")
    if last_step != -1:
        xm.master_print(f"Last step: {last_step}")
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps=total_steps, last_epoch=last_step,
                                        div_factor=cfg.div_factor, pct_start=cfg.pct_start)
    fn = Path(cfg.rst_path or '.')/f'{removesuffix(cfg.rst_name or "", ".pth")}.sched'
    if fn.exists() and not cfg.reset_opt:
        checkpoint = torch.load(fn, map_location='cpu')
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        xm.master_print(f"Restarting from previous scheduler state")
    scheduler.batchwise = True
    return scheduler


def train_fn(model, cfg, xm, epoch, para_loader, criterion, seg_crit, optimizer, scheduler, device):
    
    # initialize
    batch_start = time.perf_counter()
    model.train()
    optimizer.zero_grad()
    loss_meter = AverageMeter(xm)
    if cfg.use_aux_loss: seg_loss_meter = AverageMeter(xm)
        
    # prepare batch_tfms (on device)
    if cfg.use_batch_tfms:
        raise NotImplementedError('get aug_flags first!')
        horizontal_flip = aug_flags['horizontal_flip']
        vertical_flip = aug_flags['vertical_flip']
        p_grayscale = aug_flags['p_grayscale']
        jitter_brightness = aug_flags['jitter_brightness']
        jitter_saturation = aug_flags['jitter_saturation']
        max_rotate = aug_flags['max_rotate']
        p_rotation = 0.8 if max_rotate > 0 else 0
        diag = math.sqrt(size[0]**2 + size[1]**2)
        padding = [1 + int((diag - s) / 2) for s in size[::-1]]  # [x,y]
        left, top = padding
        height, width = size

    # training loop
    for batch_idx, batch in enumerate(para_loader, start=1):

        # extract inputs and labels (multi-core: data already on device)
        if cfg.use_aux_loss:
            inputs, masks, labels = batch
            masks = masks.to(device)
        else:
            inputs, labels = batch
        del batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        # image batch_tfms
        if cfg.use_batch_tfms:
            if size[1] > size[0]:
                # train on 90-deg rotated nybg2021 images
                inputs = inputs.transpose(-2,-1)
            if horizontal_flip and random.random() > 0.5:
                inputs = TF.hflip(inputs)
            if vertical_flip and random.random() > 0.5:
                inputs = TF.vflip(inputs)
            if p_grayscale and random.random() > p_grayscale:
                inputs = TF.rgb_to_grayscale(inputs, num_output_channels=3)  # try w/o num_output_channels
            if jitter_brightness:
                if isinstance(jitter_brightness, int):
                    mu, sigma = 1.0, jitter_brightness
                else:
                    mu = np.mean(jitter_brightness)
                    sigma = (max(jitter_brightness) - min(jitter_brightness)) * 0.28868
                brightness_factor = random.normalvariate(mu, sigma)
                inputs = TF.adjust_brightness(inputs, brightness_factor)
            if jitter_saturation:
                if isinstance(jitter_saturation, int):
                    mu, sigma = 1.0, jitter_saturation
                    saturation_factor = random.normalvariate(mu, sigma)
                else:
                    saturation_factor = random.uniform(*jitter_saturation)
                inputs = TF.adjust_saturation(inputs, saturation_factor)            
            if p_rotation and random.random() > p_rotation:
                angle = random.randint(-max_rotate, max_rotate)
                inputs = TF.pad(inputs, padding, padding_mode='reflect')
                inputs = TF.rotate(inputs, angle, resample=PIL.Image.NEAREST)
                #v0.9# inputs = TF.rotate(inputs, angle, interpolation='nearest')
                inputs = TF.crop(inputs, top, left, height, width)
        
        # forward and backward pass
        preds = model(inputs)
        if cfg.use_aux_loss:
            seg_logits, preds = preds
            seg_loss = seg_crit(seg_logits, masks)
            cls_loss = criterion(preds, labels)
            loss = cfg.seg_weight * seg_loss + (1 - cfg.seg_weight) * cls_loss
        else:
            loss = criterion(preds, labels)
        loss = loss / cfg.n_acc
        loss.backward()
        if batch_idx % cfg.n_acc == 0:
            xm.optimizer_step(optimizer, barrier=True)   # rendevouz, required for proper xmp shutdown
            optimizer.zero_grad()
            if hasattr(scheduler, 'step') and hasattr(scheduler, 'batchwise'): scheduler.step()
        
        # aggregate loss locally
        if cfg.use_aux_loss:
            # In grad accumulation, loss is scaled, but not cls_loss and seg_loss.
            loss_meter.update(cls_loss.item(), inputs.size(0))
            seg_loss_meter.update(seg_loss.item(), inputs.size(0))
        else:
            loss_meter.update(loss.item() * cfg.n_acc, inputs.size(0))
        
        # print batch_verbose information
        if cfg.batch_verbose and (batch_idx % cfg.batch_verbose == 0):
            info_strings = [
                f'        batch {batch_idx}',
                f'current_loss {loss_meter.current:.5f}']
            if cfg.use_aux_loss: 
                info_strings.append(f'seg_loss {seg_loss_meter.current:.5f}')
            info_strings.append(f'avg_loss {loss_meter.average:.5f}')
            info_strings.append(f'lr {optimizer.param_groups[-1]["lr"] / xm.xrt_world_size():7.1e}')
            info_strings.append(f'mom {optimizer.param_groups[-1]["betas"][0]:.3f}')
            info_strings.append(f'time {(time.perf_counter() - batch_start) / 60:.2f} min')
            xm.master_print(', '.join(info_strings))
            if hasattr(scheduler, 'get_last_lr'):
                assert scheduler.get_last_lr()[-1] == optimizer.param_groups[-1]['lr']
            batch_start = time.perf_counter()
    
    # scheduler step after epoch
    if hasattr(scheduler, 'step') and not hasattr(scheduler, 'batchwise'): scheduler.step()

    return loss_meter.average


def valid_fn(model, cfg, xm, epoch, para_loader, criterion, device, metrics=None):

    # initialize
    model.eval()
    loss_meter = AverageMeter(xm)
    metrics = metrics or []
    if metrics:
        # macro metrics need all predictions and labels
        all_scores = []
        all_preds = []
        all_labels = []
    
    # validation loop
    for batch_idx, (inputs, labels) in enumerate(para_loader, start=1):

        # extract inputs and labels (multi-core: ParallelLoader does this already)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.no_grad():
            if cfg.use_aux_loss:
                seg_preds, preds = model(inputs)
            else:
                preds = model(inputs)
        
        # compute local loss
        loss = criterion(preds, labels)
        loss_meter.update(loss.item(), inputs.size(0))
        
        # locally keep preds, labels for metrics (needs only device memory)
        if metrics and cfg.multilabel:
            all_scores.append(preds.detach().sigmoid())
            all_preds.append((all_scores[-1] > 0.5).to(int))
            all_labels.append(labels)
        elif metrics:
            all_scores.append(preds.detach().softmax(dim=1))  # for mAP
            all_preds.append(preds.detach().argmax(dim=1))
            all_labels.append(labels)
            
    # mesh_reduce loss
    avg_loss = loss_meter.average
    
    # mesh_reduce metrics
    avg_metrics = []
    if metrics:
        metrics_start = time.perf_counter()
        local_scores = torch.cat(all_scores)
        local_preds = torch.cat(all_preds)
        local_labels = torch.cat(all_labels)
        scores = xm.mesh_reduce('reduce_scores', local_scores, torch.cat)
        preds = xm.mesh_reduce('reduce_preds', local_preds, torch.cat)
        labels = xm.mesh_reduce('reduce_labels', local_labels, torch.cat)
        ## todo: try avoid lowering by using torch metrics instead of sklearn
        #xm.master_print("labels:", labels.cpu().numpy()[0])
        #xm.master_print("scores:", scores.cpu().numpy()[0])
        #xm.master_print("preds: ",  preds.cpu().numpy()[0])
        for m in metrics:
            avg_metrics.append(
                m(labels.cpu().numpy(), scores.cpu().numpy()) if hasattr(m, 'needs_scores') else
                m(labels.cpu().numpy(), preds.cpu().numpy())
            )
        #wall_metrics = time.perf_counter() - metrics_start
        #xm.master_print(f"Wall metrics: {xm.mesh_reduce('avg_wall_metrics', wall_metrics, max) / 60:.2f} min")
        
    return avg_loss, avg_metrics


def _mp_fn(rank, cfg, metadata, wrapped_model, serial_executor, xm, use_fold, class_column='category_id'):
    "Distributed training loop master function"

    ### Setup
    if cfg.xla:
        import torch_xla.distributed.parallel_loader as pl
        from catalyst.data import DistributedSamplerWrapper
        from torch.utils.data.distributed import DistributedSampler
        loader_prefetch_size = 1
        device_prefetch_size = 1
    
    # Data samplers, class-weighted metrics
    train_labels = metadata.loc[~ metadata.is_valid, class_column]
    valid_labels = metadata.loc[  metadata.is_valid, class_column]

    train_tfms = get_tfms(cfg, mode='train')
    test_tfms = get_tfms(cfg, mode='test')
    tensor_tfms = None
    
    if cfg.use_aux_loss:
        ds_train = MySiimCovidAuxDataset(metadata.loc[~ metadata.is_valid], cfg, mode='train', 
                                         transform=train_tfms, tensor_transform=tensor_tfms)
    else:
        ds_train = ImageDataset(metadata.loc[~ metadata.is_valid], cfg, mode='train', 
                                         transform=train_tfms, tensor_transform=tensor_tfms)
        
    ds_valid = ImageDataset(metadata.loc[  metadata.is_valid], cfg, mode='valid',
                                         transform=test_tfms, tensor_transform=tensor_tfms)

    #xm.master_print("train_tfms:")
    #xm.master_print(ds_train.transform)
    #xm.master_print("test_tfms:")
    #xm.master_print(ds_valid.transform)
    
    if cfg.do_class_sampling:
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
        xm.master_print(f"Sampling {int(n_examples * frac)} out of {n_examples} examples")
        weighted_train_sampler = WeightedRandomSampler(
            sample_weights, int(n_examples*cfg.frac), replacement=True)
        
        # Wrap it in a DistributedSamplerWrapper
        train_sampler = (DistributedSamplerWrapper(weighted_train_sampler,
                                                   num_replicas = xm.xrt_world_size(),
                                                   rank         = xm.get_ordinal(),
                                                   shuffle      = False) if cfg.xla else 
                         weighted_train_sampler)
    
    elif xm.xrt_world_size() > 1:
        train_sampler = DistributedSampler(ds_train,
                                           num_replicas = xm.xrt_world_size(),
                                           rank         = xm.get_ordinal(),
                                           shuffle      = True)
    
    else:
        train_sampler = None
    
    valid_sampler = (DistributedSampler(ds_valid,
                                        num_replicas = xm.xrt_world_size(),
                                        rank         = xm.get_ordinal(),
                                        shuffle      = False) if xm.xrt_world_size() > 1 else
                     None)
    
    # Dataloaders
    train_loader = DataLoader(ds_train,
                              batch_size  = cfg.bs, 
                              sampler     = train_sampler,
                              num_workers = 0 if xm.xrt_world_size() > 1 else cpu_count(),
                              pin_memory  = True,
                              drop_last   = True,
                              shuffle     = False if train_sampler else True)
    valid_loader = DataLoader(ds_valid, 
                              batch_size  = cfg.bs, 
                              sampler     = valid_sampler, 
                              num_workers = 0 if xm.xrt_world_size() > 1 else cpu_count(),
                              pin_memory  = True,
                             ) 
    
    # Send model to device
    device = xm.xla_device()
    model = wrapped_model.to(device)
    
    # Criterion, Metrics
    criterion = nn.BCEWithLogitsLoss() if cfg.multilabel else nn.CrossEntropyLoss()
    if cfg.use_aux_loss:
        from segmentation_models_pytorch.losses.dice import DiceLoss
    seg_crit = DiceLoss('binary') if cfg.use_aux_loss else None
    micro_f1 = partial(f1_score, average='micro')
    micro_f1.__name__ = 'F1'
    macro_f1 = partial(f1_score, average='macro', labels=valid_labels.values)
    macro_f1.__name__ = 'F1'
    acc = accuracy_score
    acc.__name__ = 'acc'
    ap = multiclass_average_precision_score
    ap.__name__ = 'acc'
    ap.needs_scores = True
    lap = label_ranking_average_precision_score
    lap.__name__ = 'lAP'
    lap.needs_scores = True
    map = partial(val_map, xm=xm)
    map.__name__ = 'mAP'
    map.needs_scores = True
    metrics = (
        []                     if len(ds_valid) == 0 else
        [ap,  micro_f1, map]   if cfg.multilabel else
        [acc, macro_f1, map]
    )
    
    # Scale LRs
    lr_head, lr_bn, lr_body = cfg.lr_head, cfg.lr_bn, cfg.lr_body
    scaled_lr_bn = lr_bn * xm.xrt_world_size()
    scaled_lr_body = lr_body * xm.xrt_world_size()
    scaled_lr_head = lr_head * xm.xrt_world_size()

    # Parameter Groups
    use_parameter_groups = False if lr_head == lr_bn == lr_body else True
    if use_parameter_groups:
        xm.master_print(f"Using parameter groups. lr_head={lr_head}, lr_body={lr_body}, lr_bn={lr_bn}")
        parameter_groups = {
            'body': (p for name, p in model.body.named_parameters() if not is_bn(name)),
            'head': model.head.parameters(),
            'bn':   (p for name, p in model.body.named_parameters() if is_bn(name)),
        }
        max_lrs = {'body': scaled_lr_body, 'head': scaled_lr_head, 'bn': scaled_lr_bn}
        params = [{'params': parameter_groups[g], 'lr': max_lrs[g]} 
                  for g in parameter_groups.keys()]
        max_lrs = list(max_lrs.values())
    else:
        max_lrs = scaled_lr_head
        params = model.parameters()

    # Optimizer
    optimizer = (
        optim.AdamW(params, lr=scaled_lr_head, betas=cfg.betas, weight_decay=cfg.wd) if cfg.optimizer == 'AdamW' else
        optim.Adam(params, lr=scaled_lr_head, betas=cfg.betas)                       if cfg.optimizer == 'Adam' else
        optim.SGD(params, lr=scaled_lr_head, momentum=cfg.betas[0], dampening=1-cfg.betas[1]))
    rst_epoch = 0
    if cfg.rst_name:
        fn = Path(cfg.rst_path)/f'{removesuffix(cfg.rst_name, ".pth")}.opt'
        if fn.exists() and not cfg.reset_opt:
            checkpoint = torch.load(fn, map_location='cpu')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            rst_epoch = checkpoint['epoch'] + 1
            xm.master_print(f"Restarting from previous opt state")
    
    # Scheduler
    if cfg.one_cycle:
        scheduler = get_one_cycle_scheduler(optimizer, ds_train, max_lrs, cfg, xm, rst_epoch, train_loader)
    elif cfg.reduce_on_plateau:
        # ReduceLROnPlateau must be called after validation
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                                                   patience=5, verbose=True, eps=1e-6)
    elif isinstance(cfg.step_lr_after, int) and cfg.step_lr_factor > 0:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.step_lr_after, 
                                        gamma=cfg.step_lr_factor)
    elif hasattr(cfg.step_lr_after, '__iter__') and cfg.step_lr_factor > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=cfg.step_lr_after,
                                             gamma=cfg.step_lr_factor)
    else: 
        scheduler = None
    if scheduler: 
        xm.master_print(f"Scheduler: {scheduler.__class__.__name__}")
        xm.master_print(f"""Initial lrs: {', '.join(f'{p["lr"]/xm.xrt_world_size():7.2e}' for p in optimizer.param_groups)}""")
        xm.master_print(f"Max lrs:     {', '.join(f'{lr/xm.xrt_world_size():7.2e}' for lr in listify(max_lrs))}")
    
    # Maybe freeze body
    if hasattr(model, 'body') and lr_body == 0:
        xm.master_print("Freezing body of pretrained model")
        for n, p in model.body.named_parameters():
            if not is_bn(n):
                p.requires_grad = False

    model_name = f'{cfg.name}_fold{use_fold}'
    xm.master_print(f'Checkpoints will be saved as {cfg.out_dir}/{model_name}_ep*')

    

    ### Training Loop ---------------------------------------------------------
    
    lrs = []
    train_losses = []
    valid_losses = []
    metrics_dicts = []
    minutes = []
    best_valid_loss = 10
    best_model_score = -np.inf
    epoch_summary_header = ''.join(['epoch   ', ' train_loss ', ' valid_loss ',
        ' '.join([f'{m.__name__:^8}' for m in metrics]),
        '   lr    ', 'min_train  min_total'])
    xm.master_print("\n", epoch_summary_header)
    xm.master_print("=" * (len(epoch_summary_header) + 2))
        
    for epoch in range(rst_epoch, rst_epoch + cfg.epochs):
        
        # Data for verbose info
        scaled_lr = optimizer.param_groups[-1]["lr"]
        if hasattr(scheduler, 'get_last_lr'):
            assert scheduler.get_last_lr()[-1] == optimizer.param_groups[-1]["lr"]
        epoch_start = time.perf_counter()
        
        # Update train_loader shuffling
        if hasattr(train_loader.sampler, 'set_epoch'): train_loader.sampler.set_epoch(epoch)
        
        # Training
        train_start = time.perf_counter()        
        para_loader = (pl.ParallelLoader(train_loader, [device],
                                         loader_prefetch_size=loader_prefetch_size,
                                         device_prefetch_size=device_prefetch_size
                                         ).per_device_loader(device) if cfg.xla else
                       train_loader)
        
        train_loss = train_fn(model, cfg, xm,
                              epoch       = epoch + 1, 
                              para_loader = para_loader,
                              criterion   = criterion,
                              seg_crit    = seg_crit,
                              optimizer   = optimizer, 
                              scheduler   = scheduler,
                              device      = device)
        
        # Validation
        valid_start = time.perf_counter()
        if cfg.train_on_all:
            valid_loss, valid_metrics = 0, []
        else:            
            para_loader = (pl.ParallelLoader(valid_loader, [device],
                                             loader_prefetch_size=loader_prefetch_size,
                                             device_prefetch_size=device_prefetch_size
                                             ).per_device_loader(device) if cfg.xla else
                           valid_loader)

            valid_loss, valid_metrics = valid_fn(model, cfg, xm,
                                                 epoch       = epoch + 1, 
                                                 para_loader = para_loader,
                                                 criterion   = criterion, 
                                                 device      = device,
                                                 metrics     = metrics)
        metrics_dict = {m.__name__: val for m, val in zip(metrics, valid_metrics)}
        last_lr = optimizer.param_groups[-1]["lr"] if hasattr(scheduler, 'batchwise') else scaled_lr
        avg_lr = 0.5 * (scaled_lr + last_lr) / xm.xrt_world_size()

        # Print epoch summary
        epoch_summary_strings = [f'{epoch + 1:>2} / {rst_epoch + cfg.epochs:<2}']          # ep/epochs
        epoch_summary_strings.append(f'{train_loss:10.5f}')                                # train_loss
        epoch_summary_strings.append(f'{valid_loss:10.5f}')                                # valid_loss
        for val in valid_metrics:                                                          # metrics
            epoch_summary_strings.append(f'{val:7.5f}')
        epoch_summary_strings.append(f'{avg_lr:7.1e}')                                     # lr
        epoch_summary_strings.append(f'{(valid_start - train_start) / 60:7.2f}')           # Wall train
        epoch_summary_strings.append(f'{(time.perf_counter() - train_start) / 60:7.2f}')   # Wall total
        xm.master_print('  '.join(epoch_summary_strings))
        
        # Save weights, optimizer state, scheduler state
        # Note: xm.save must not be inside an if statement that may validate differently on
        # different TPU cores. Reason: rendezvous inside them will hang if any core
        # does not arrive at the rendezvous.
        model_score = (metrics_dict[cfg.save_best] if cfg.save_best and cfg.save_best in metrics_dict else
                       -valid_loss)
        if model_score > best_model_score or not cfg.save_best:
            if cfg.save_best:
                best_model_score = model_score
                #xm.master_print(f'{cfg.save_best or "valid_loss"} improved.')
            
            #xm.master_print(f'saving {model_name}_ep{epoch+1}.pth ...')
            xm.save(model.state_dict(), f'{cfg.out_dir}/{model_name}_ep{epoch+1}.pth')
            
            #xm.master_print(f'saving {model_name}_ep{epoch+1}.opt ...')
            xm.save({'optimizer_state_dict': optimizer.state_dict(),
                     'epoch': epoch}, f'{cfg.out_dir}/{model_name}_ep{epoch+1}.opt')
            
            if hasattr(scheduler, 'state_dict'):
                #xm.master_print(f'saving {model_name}_ep{epoch+1}.sched ...')
                xm.save({'scheduler_state_dict': {
                    k: v for k, v in scheduler.state_dict().items() if k != 'anneal_func'}},
                        f'{cfg.out_dir}/{model_name}_ep{epoch+1}.sched')
                        
        # Save losses, metrics
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        metrics_dicts.append(metrics_dict)
        lrs.append(avg_lr)
        minutes.append((time.perf_counter() - train_start) / 60)
    
    if xm.xrt_world_size() > 1:
        serial_executor.run(lambda: save_metrics(train_losses, valid_losses, metrics_dicts, 
                                                 lrs, minutes, rst_epoch, use_fold, cfg.out_dir))
    else:
        save_metrics(train_losses, valid_losses, metrics_dicts, lrs, minutes, rst_epoch, use_fold, cfg.out_dir)

    
def save_metrics(train_losses, valid_losses, metrics_dicts, lrs, minutes, rst_epoch, fold, out_dir):
    df = pd.DataFrame({"train_loss": train_losses, "valid_loss": valid_losses})
    df = pd.concat([df, pd.DataFrame(metrics_dicts)], axis=1)
    df['lr'] = lrs
    df['Wall'] = minutes
    df['epoch'] = df.index + rst_epoch + 1
    df.set_index('epoch', inplace=True)
    df.to_json(f'{out_dir}/metrics_fold{fold}.json')