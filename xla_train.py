from pathlib import Path
from contextlib import nullcontext
from future import removesuffix
import time
import math

import numpy as np
from metrics import get_tm_metrics, is_listmetric, AverageMeter
from torch_data import get_dataloaders, get_fakedata_loaders
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.schedulers import get_one_cycle_scheduler, maybe_step
from utils.general import listify
from models import is_bn
from torch import FloatTensor, LongTensor
from torchvision.io import encode_jpeg, decode_jpeg


torch.set_default_tensor_type('torch.FloatTensor')


def train_fn(model, cfg, xm, dataloader, criterion, seg_crit, optimizer, scheduler, device):

    # initialize
    batch_start = time.perf_counter()
    model.train()
    optimizer.zero_grad()
    loss_meter = AverageMeter(xm)
    if cfg.use_aux_loss: seg_loss_meter = AverageMeter(xm)
    if cfg.loss_weights is not None:
        weighted_loss_meters = [AverageMeter(xm) for _ in cfg.loss_weights]
    
    # setup AMP
    if cfg.dtype == 'float16' and not (cfg.gpu or cfg.xla):
        xm.master_print(f'WARNING: dtype float16 not supported on CPU, changing to bfloat16.')
        cfg.dtype = 'bfloat16'
    amp_context = (
        nullcontext() if cfg.dtype == 'float32' else
        torch.amp.autocast('cuda' if cfg.gpu else 'cpu', dtype=getattr(torch, cfg.dtype)))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.gpu and cfg.dtype.endswith('float16'))

    # prepare batch_tfms (on device)
    if cfg.use_batch_tfms:
        #import torchvision.transforms as TT
        import torchvision.transforms.functional as TF
        # antialias is currently stalling xla compiling and False by default, but
        # will default to True in v0.17
        cfg.antialias = cfg.antialias or False

        # TT.Random* tfms disable fast TPU execution!
        #batch_tfms = torch.nn.Sequential(
        #    #TT.RandomRotation(degrees=5),
        #    TT.GaussianBlur(kernel_size=5),  # OK
        #    TT.RandomHorizontalFlip(p=0.5),  # issue
        #    #TT.RandomGrayscale(0.2),
        #    TT.RandomResizedCrop(cfg.size, scale=(0.5, 1.0)),  # issue
        #    TT.Resize(cfg.size),  # OK
        #    )
        #batch_tfms = TT.Compose(  # does not help with random tfms
        #    TT.RandomHorizontalFlip(),
        #    TT.Resize(cfg.size),
        #)
        #batch_tfms = batch_tfms.to(device)  # same or worse with random tfms

    #if cfg.use_batch_tfms:
    #    raise NotImplementedError('get aug_flags first!')
    #    xm.master_print("aug_flags:")
    #    for k, v in aug_flags.items():
    #        xm.master_print(f'{k:<30} {v}')
    #    horizontal_flip = aug_flags['horizontal_flip']
    #    vertical_flip = aug_flags['vertical_flip']
    #    p_grayscale = aug_flags['p_grayscale']
    #    jitter_brightness = aug_flags['jitter_brightness']
    #    jitter_saturation = aug_flags['jitter_saturation']
    #    max_rotate = aug_flags['max_rotate']
    #    p_rotation = 0.8 if max_rotate > 0 else 0
    #    diag = math.sqrt(cfg.size[0] ** 2 + cfg.size[1] ** 2)
    #    padding = [1 + int((diag - s) / 2) for s in cfg.size[::-1]]  # [x,y]
    #    left, top = padding
    #    height, width = cfg.size

    # training loop
    n_iter = len(dataloader)
    iterable = range(n_iter) if (cfg.fake_data == 'on_device') else dataloader
    #sample_iterator = iter(dataloader) if cfg.use_batch_tfms else None
    ### DEBUG
    timers = [0, 0, 0, 0, 0]

    for batch_idx, batch in enumerate(iterable, start=1):

        # extract inputs and labels
        if cfg.fake_data == 'on_device':
            inputs, labels = (
                torch.ones(cfg.bs, 3, *cfg.size, dtype=torch.uint8, device=device) * 128,
                torch.exp(torch.randn(cfg.bs, 3, device=device) * 0.6))
                #torch.zeros(cfg.bs, dtype=torch.int64, device=device))
        #elif cfg.use_batch_tfms:
            # resize and collate images, labels
            #samples = [next(sample_iterator) for _ in range(cfg.bs)]
            #for s in samples:
            #    assert s[0].device == device
            #    assert s[1].device == device
            #inputs = torch.stack([pre_size(s[0]) for s in samples])
            #labels = torch.stack([s[1] for s in samples])
            #del samples

        #    inputs, labels = [], []
        #    for _ in range(cfg.bs):
        #        s = next(sample_iterator)
        #        inputs.append(TF.resize(s[0], cfg.size, InterpolationMode('nearest')))
        #        labels.append(s[1])
        #    inputs = torch.stack(inputs)
        #    labels = torch.stack(labels)

            #assert inputs.shape == (cfg.bs, 3, *cfg.size)
            #assert labels.shape == (cfg.bs,)
            #xm.master_print(inputs.device, labels.device)

        elif cfg.filetype == 'tfds':
            inputs, labels = FloatTensor(batch[0]['inp1']), LongTensor(batch[0]['inp2'])
            inputs = inputs.permute((0, 3, 1, 2))  #.contiguous()  # mem? speed?
        elif cfg.use_aux_loss:
            inputs, masks, labels = batch
            masks = masks.to(device)
        else:
            inputs, labels = batch
        del batch
        #xm.master_print("inputs:", type(inputs), inputs.shape, inputs.dtype, inputs.device)
        #assert inputs.shape == (cfg.bs, 3, *cfg.size), f'wrong inputs shape: {inputs.shape}'
        #assert labels.shape == (cfg.bs,), f'wrong labels shape: {labels.shape}'
        #print(f"rank {xm.get_ordinal()} labels: {labels}")

        # send to device(s) if still on CPU (device_loaders do this automatically)
        if True:
            inputs = inputs.to(device)
            labels = labels.to(device)
        else:
            inputs, labels = (
                torch.zeros(cfg.bs, 3, *cfg.size, device=device),
                torch.zeros(cfg.bs, dtype=torch.int64, device=device))

        # image batch_tfms
        if cfg.use_batch_tfms:
            #inputs = batch_tfms(inputs)  # all TT.Random* tfms break fast tpu execution

            # RandomResizedCrop (breaks fast TPU exe if tensor shapes depend on torch.rand())
            #size = torch.tensor(cfg.size)
            #height_width = ((0.5 + 0.5 * torch.rand(2)) * size).to(torch.int)  # bad
            #top, left = ((size - height_width) * torch.rand(2)).to(torch.int)  # bad
            #height, width = height_width
            #height = int((scale_min + (1 - scale_min) * torch.rand(1)) * cfg.size[0])  # bad
            #width = int((scale_min + (1 - scale_min) * torch.rand(1)) * cfg.size[1])  # bad
            #inputs = TF.crop(inputs, 10, 10, int(0.9 * cfg.size[0]), int(0.9 * cfg.size[1]))  # OK
            #scale_min = 0.6
            #top = 10
            #left = 10
            #r0, r1 = 0.33, 0.74  # OK
            #height = int((scale_min + (1 - scale_min) * r0) * cfg.size[0])
            #width = int((scale_min + (1 - scale_min) * r1) * cfg.size[1])
            #inputs = TF.resized_crop(inputs, top, left, height, width, cfg.size)

            if cfg.DEBUG:
                assert inputs.dtype == torch.uint8, f'unexpected inputs dtype {inputs.dtype}'

            if cfg.curve == 'gamma':
                # unpack labels: (gamma, abs_log_gamma, rel_log_gamma, rnd_factor, bp_shift)
                if cfg.noise_level or cfg.random_blackpoint_shift:
                    labels, rnd_vars = labels[:, :9], labels[:, 9:]
                    if cfg.noise_level: rnd_factor = rnd_vars[:, 0]
                    if cfg.random_blackpoint_shift: bp_shift = rnd_vars[:, -3:]
                gamma, abs_log_gamma, rel_log_gamma = labels[:, :3], labels[:, 3:6], labels[:, 6:]
                if cfg.loss_weights is None:
                    labels = gamma
            elif (cfg.curve == 'free'):
                # unpack labels in reverse order
                if cfg.sharpness_augment:
                    # use same sharpness on all channels
                    labels, rnd_sharpness = labels[:, :, :-1], labels[:, 0, -1]
                if cfg.add_jpeg_artifacts:
                    # use same jpeg_quality on all channels
                    labels, jpeg_quality = labels[:, :, :-1], labels[:, 0, -1]
                if cfg.predict_inverse and cfg.noise_level:
                    # split labels into (labels+rnd_noise, tfms)
                    labels, tfms = labels[:, :, :257], labels[:, :, 257:]
                elif cfg.predict_inverse:
                    # split labels into (target, tfms)
                    labels, tfms = labels[:, :, :256], labels[:, :, 256:]
                else:
                    tfms = labels

            if (cfg.curve == 'beta') and cfg.curve_tfm_on_device:
                # labels: (N, C, 3), curves: (N, C, 256)
                labels, curves = labels[:, :, :3], labels[:, :, 3:]

                if False:
                    # (a): map curves via elementwise assignment
                    inputs = inputs.to(torch.int64)
                    transformed = torch.empty(size=inputs.size(), dtype=curves.dtype, device=inputs.device) 
                    for i, img_curves in enumerate(curves):
                        for j, curve in enumerate(img_curves):
                            transformed[i, j, :, :] = curve[inputs[i, j, :, :]]
                    inputs = transformed
                    del transformed

                else:
                    # (b): map curves with torch.gather (slightly faster)
                    # curves must be expanded to have same shape as inputs (except dim=2)
                    expanded_curves = curves[..., None].expand(-1, -1, -1, inputs.size(-1))
                    if cfg.add_uniform_noise:
                        # add uniform noise to mask uint8 discretization
                        inputs_plus_one = (inputs.to(torch.int64) + 1).clamp(0, 255)
                    inputs = torch.gather(expanded_curves, dim=2, index=inputs.to(torch.int64))
                    if cfg.add_uniform_noise:
                        inputs_plus_one = torch.gather(expanded_curves, dim=2, index=inputs_plus_one)
                        noise_range = inputs_plus_one - inputs
                        noise = torch.rand_like(inputs) * noise_range
                        inputs += noise

            elif cfg.curve == 'free':
                # map tfms (N, C, 256) with torch.gather
                # curves must be expanded to have same shape as inputs (except dim=2)
                expanded_curves = tfms[..., None].expand(-1, -1, -1, inputs.size(-1))
                if cfg.add_uniform_noise:
                    # add uniform noise to mask uint8 discretization
                    inputs_plus_one = (inputs.to(torch.int64) + 1).clamp(0, 255)
                inputs = torch.gather(expanded_curves, dim=2, index=inputs.to(torch.int64))
                if cfg.add_uniform_noise:
                    inputs_plus_one = torch.gather(expanded_curves, dim=2, index=inputs_plus_one)
                    noise_range = inputs_plus_one - inputs
                    noise = torch.rand_like(inputs) * noise_range
                    inputs += noise

            else:
                inputs = inputs.float()

            if cfg.add_uniform_noise and (cfg.curve == 'gamma'):
                inputs = (inputs + torch.rand_like(inputs)).clamp(max=255)

            if cfg.curve != 'free':
                inputs /= 255

            if cfg.curve == 'gamma':
                inputs = torch.pow(inputs, gamma[:, :, None, None])  # channel first

            if cfg.sharpness_augment:
                # as augmentation, sharpen should happen before adding JPEG noise
                #rnd_test = 2.0 * torch.rand(1)  # OK on single core
                inputs = TF.adjust_sharpness(inputs, rnd_sharpness[0])  # batch-wise sharpness
                #xm.master_print("rnd_sharpness, rnd_test:", rnd_sharpness[0], rnd_test)

            if cfg.add_jpeg_artifacts:  # slow!
                inputs = (inputs.clamp(0, 1) * 255).to(torch.uint8)
                B, C, H, W = inputs.shape

                # albumentation: HWC-numpy array
                #inputs = inputs.permute(0, 2, 3, 1).reshape(B * H, W, C)
                #inputs = torch.tensor(ImageCompression(50, 100, p=1)(image=inputs.numpy())['image'])
                #inputs = inputs.reshape(B, H, W, C).permute(0, 3, 1, 2)

                # torchvision: CHW tensor
                inputs = inputs.transpose(0, 1).reshape(C, B * H, W)
                #xm.master_print("jpeg_quality:", jpeg_quality[0])
                jpeg = encode_jpeg(inputs.cpu(), jpeg_quality[0])  # batch-wise quality
                inputs = decode_jpeg(jpeg).to(device)  # jpeg tensor must be on CPU!
                inputs = inputs.reshape(C, B, H, W).transpose(1, 0)
                inputs = inputs.float() / 255

            inputs = TF.resize(inputs, cfg.size, antialias=cfg.antialias)

            # RandomHorizontalFlip (OK)
            if torch.rand(1) > 0.5:
                inputs = TF.hflip(inputs)

            #if cfg.size[1] > cfg.size[0]:
            #    # train on 90-deg rotated nybg2021 images
            #    inputs = inputs.transpose(-2,-1)
            #if horizontal_flip and random.random() > 0.5:
            #    inputs = TF.hflip(inputs)
            #if vertical_flip and random.random() > 0.5:
            #    inputs = TF.vflip(inputs)
            #if p_grayscale and random.random() > p_grayscale:
            #    inputs = TF.rgb_to_grayscale(inputs, num_output_channels=3)
            #    # try w/o num_output_channels
            #if jitter_brightness:
            #    if isinstance(jitter_brightness, int):
            #        mu, sigma = 1.0, jitter_brightness
            #    else:
            #        mu = np.mean(jitter_brightness)
            #        sigma = (max(jitter_brightness) - min(jitter_brightness)) * 0.28868
            #    brightness_factor = random.normalvariate(mu, sigma)
            #    inputs = TF.adjust_brightness(inputs, brightness_factor)
            #if jitter_saturation:
            #    if isinstance(jitter_saturation, int):
            #        mu, sigma = 1.0, jitter_saturation
            #        saturation_factor = random.normalvariate(mu, sigma)
            #    else:
            #        saturation_factor = random.uniform(*jitter_saturation)
            #    inputs = TF.adjust_saturation(inputs, saturation_factor)
            #if p_rotation and random.random() > p_rotation:
            #    angle = random.randint(-max_rotate, max_rotate)
            #    inputs = TF.pad(inputs, padding, padding_mode='reflect')
            #    inputs = TF.rotate(inputs, angle, resample=0)
            #    inputs = TF.crop(inputs, top, left, height, width)

        # Random Noise
        if cfg.noise_level:
            # unpack labels, use same noise_level on all channels
            labels, rnd_factor = labels[:, :, :-1], labels[:, 0, -1]
            inputs += cfg.noise_level * rnd_factor[:, None, None, None] * torch.randn_like(inputs)
            inputs = inputs.clamp(0.0, 1.0)

        if cfg.curve and (cfg.curve == 'free'):
            labels = labels.reshape(labels.shape[0], -1)

        # forward and backward pass
        perform_optimizer_step = (batch_idx % cfg.n_acc == 0)
        if cfg.use_ddp: model.require_backward_grad_sync = perform_optimizer_step
        ### DEBUG
        t0 = time.perf_counter()
        with amp_context:
            preds = model(inputs, labels) if model.requires_labels else model(inputs)
            ### DEBUG
            t1 = time.perf_counter()
            
            # calculate loss
            if cfg.use_aux_loss:
                seg_logits, preds = preds
                seg_loss = seg_crit(seg_logits, masks)
                cls_loss = criterion(preds, labels)
                loss = cfg.seg_weight * seg_loss + (1 - cfg.seg_weight) * cls_loss
            if cfg.loss_weights is not None:
                if cfg.curve == 'gamma':
                    _preds = [preds - torch.mean(preds, dim=1, keepdim=True), preds]
                    _labels = [rel_log_gamma, abs_log_gamma]
                elif cfg.curve == 'beta':
                    _preds = preds.reshape(preds.shape[0], -1, len(cfg.loss_weights)).transpose(0, 2)
                    _labels = labels.reshape(labels.shape[0], -1, len(cfg.loss_weights)).transpose(0, 2)
                else:
                    _preds, _labels = preds, labels

                weighted_losses = [w * criterion(p, l) for w, p, l in zip(cfg.loss_weights, _preds, _labels)]
                loss = sum(weighted_losses)

                # DEBUG nan loss
                #if any([torch.isnan(l.detach()) for l in weighted_losses]):
                #    xm.master_print("NaN loss detected.")
                #    xm.master_print("losses:", weighted_losses)
                #    xm.master_print("labels:", _labels)
                #    xm.master_print("preds:", _preds)

            else:
                loss = criterion(preds, labels)

            loss = loss / cfg.n_acc  # grads accumulate as 'sum' but loss reduction is 'mean'
            ### DEBUG
            t2 = time.perf_counter()
            scaler.scale(loss).backward()  # loss scaling in mixed-precision training
            ### DEBUG
            t3 = time.perf_counter()
            t4 = t3

        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        if perform_optimizer_step:
            if cfg.xla:
                xm.optimizer_step(optimizer, barrier=True)  # rendevouz, required for proper xmp shutdown
            else:
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ### DEBUG
            t4 = time.perf_counter()
            if hasattr(scheduler, 'step') and hasattr(scheduler, 'batchwise'):
                maybe_step(scheduler, xm)

        # aggregate loss locally
        if cfg.use_aux_loss:
            # loss components cls_loss, seg_loss were not divided by n_acc.
            loss_meter.update(cls_loss.item(), inputs.size(0))
            seg_loss_meter.update(seg_loss.item(), inputs.size(0))
        else:
            # undo "loss /= n_acc" because loss_meter reduction is 'mean'
            loss_meter.update(loss.item() * cfg.n_acc, inputs.size(0))  # 1 aten/iter, but no performance drop
            #loss_meter.update(loss.detach() * cfg.n_acc, inputs.size(0))  # recursion!
            #xm.add_step_closure(loss_meter.update, args=(loss.item(), cfg.n_acc * inputs.size(0)))  # recursion!
        if cfg.loss_weights is not None:
            for i, m in enumerate(weighted_loss_meters):
                m.update(weighted_losses[i].item() * cfg.n_acc, inputs.size(0))

        # print batch_verbose information
        if cfg.batch_verbose and (batch_idx % cfg.batch_verbose == 0):
            info_strings = [
                f'        batch {batch_idx} / {n_iter}',
                f'current_loss {loss_meter.current:.5f}']
            if cfg.use_aux_loss:
                info_strings.append(f'seg_loss {seg_loss_meter.current:.5f}')
            if cfg.loss_weights is not None:
                info_strings.extend([f'{m.current:.5f}' for m in weighted_loss_meters])
            info_strings.append(f'avg_loss {loss_meter.average:.5f}')
            info_strings.append(f'lr {optimizer.param_groups[-1]["lr"]:7.1e}')
            info_strings.append(f'mom {optimizer.param_groups[-1]["betas"][0]:.3f}')
            info_strings.append(f'time {(time.perf_counter() - batch_start) / 60:.2f} min')
            xm.master_print(', '.join(info_strings))
            if hasattr(scheduler, 'get_last_lr'):
                current_lr = optimizer.param_groups[-1]['lr']
                assert scheduler.get_last_lr()[-1] == current_lr, f'scheduler: {scheduler.get_last_lr()[-1]}, opt: {current_lr}'
            batch_start = time.perf_counter()
        if cfg.DEBUG and batch_idx == 1:
            xm.master_print(f"train inputs: {inputs.shape}, value range: {inputs.min():.2f} ... {inputs.max():.2f}")
            #break
        ### DEBUG
        t5 = time.perf_counter()
        timers[0] += t1 - t0  # forward
        timers[1] += t2 - t1  # loss
        timers[2] += t3 - t2  # scale loss & backward
        timers[3] += t4 - t3  # opt step
        timers[4] += t5 - t4  # rest
    xm.master_print("Timings:")
    for name, value in zip('forward loss backward opt_step rest'.split(), timers):
        xm.master_print(f"    {name:<10} {value}")
    xm.master_print("")

    # scheduler step after epoch
    if hasattr(scheduler, 'step') and not hasattr(scheduler, 'batchwise'):
        maybe_step(scheduler, xm)

    return loss_meter.average


def valid_fn(model, cfg, xm, dataloader, criterion, device, metrics=None):

    # initialize
    model.eval()
    if not cfg.pudae_valid:
        loss_meter = AverageMeter(xm)
    if cfg.loss_weights is not None:
        weighted_loss_meters = [AverageMeter(xm) for _ in cfg.loss_weights]
    if metrics: 
        metrics.to(device)

    if cfg.use_batch_tfms:
        import torchvision.transforms.functional as TF

    # validation loop
    n_iter = len(dataloader)
    iterable = range(n_iter) if (cfg.fake_data == 'on_device') else dataloader

    for batch_idx, batch in enumerate(iterable, start=1):

        # extract inputs and labels
        if cfg.fake_data == 'on_device':
            inputs, labels = (
                torch.ones(cfg.bs, 3, *cfg.size, dtype=torch.uint8, device=device) * 128,
                torch.exp(torch.randn(cfg.bs, 3, device=device) * 0.6))
                #torch.zeros(cfg.bs, dtype=torch.int64, device=device))
        elif cfg.filetype == 'tfds':
            inputs, labels = FloatTensor(batch[0]['inp1']), LongTensor(batch[0]['inp2'])
            inputs = inputs.permute((0, 3, 1, 2))  #.contiguous()  # mem? speed?
        else:
            inputs, labels = batch

        # send to device(s) if still on CPU (device_loaders do this automatically)
        if True:
            inputs = inputs.to(device)
            labels = labels.to(device)
        else:
            inputs, labels = (
                torch.zeros(cfg.bs, 3, *cfg.size, device=device),
                torch.zeros(cfg.bs, dtype=torch.int64, device=device))

        if cfg.use_batch_tfms:
            if cfg.curve == 'gamma':
                # unpack labels: (gamma, abs_log_gamma, rel_log_gamma, rnd_factor, bp_shift)
                if cfg.noise_level or cfg.random_blackpoint_shift:
                    labels, rnd_vars = labels[:, :9], labels[:, 9:]
                    if cfg.noise_level: rnd_factor = rnd_vars[:, 0]
                    if cfg.random_blackpoint_shift: bp_shift = rnd_vars[:, -3:]
                gamma, abs_log_gamma, rel_log_gamma = labels[:, :3], labels[:, 3:6], labels[:, 6:]
            elif (cfg.curve == 'free'):
                # unpack labels in reverse order
                if cfg.sharpness_augment:
                    # use same sharpness on all channels
                    labels, rnd_sharpness = labels[:, :, :-1], labels[:, 0, -1]
                if cfg.add_jpeg_artifacts:
                    # use same jpeg_quality on all channels
                    labels, jpeg_quality = labels[:, :, :-1], labels[:, 0, -1]
                if cfg.predict_inverse and cfg.noise_level:
                    # split labels into (labels+rnd_noise, tfms)
                    labels, tfms = labels[:, :, :257], labels[:, :, 257:]
                elif cfg.predict_inverse:
                    # split labels into (target, tfms)
                    labels, tfms = labels[:, :, :256], labels[:, :, 256:]
                else:
                    tfms = labels
                #for i, tfm in enumerate(tfms):
                #    xm.master_print(f"tfm{i:03d}:", tfm.min().item(), tfm.mean().item(), tfm.max().item())

            if (cfg.curve == 'beta') and cfg.curve_tfm_on_device:
                # labels: (N, C, 3), curves: (N, C, 256)
                labels, curves = labels[:, :, :3], labels[:, :, 3:]
                expanded_curves = curves[..., None].expand(-1, -1, -1, inputs.size(-1))
                inputs = torch.gather(expanded_curves, dim=2, index=inputs.to(torch.int64))
            elif cfg.curve == 'free':
                # map tfms (N, C, 256) with torch.gather
                # curves must be expanded to have same shape as inputs (except dim=2)
                expanded_curves = tfms[..., None].expand(-1, -1, -1, inputs.size(-1))
                if cfg.add_uniform_noise:
                    # add uniform noise to mask uint8 discretization
                    inputs_plus_one = (inputs.to(torch.int64) + 1).clamp(0, 255)
                inputs = torch.gather(expanded_curves, dim=2, index=inputs.to(torch.int64))
                if cfg.add_uniform_noise:
                    inputs_plus_one = torch.gather(expanded_curves, dim=2, index=inputs_plus_one)
                    noise_range = inputs_plus_one - inputs
                    noise = torch.rand_like(inputs) * noise_range
                    inputs += noise
            else:
                inputs = inputs.float()

            if cfg.curve != 'free':
                inputs /= 255

            if cfg.curve == 'gamma':
                inputs = torch.pow(inputs, gamma[:, :, None, None])  # channel first

            if cfg.sharpness_augment:
                # as augmentation, sharpen should happen before adding JPEG noise
                #rnd_test = 2.0 * torch.rand(1)  # OK on single core
                inputs = TF.adjust_sharpness(inputs, rnd_sharpness[0])  # batch-wise sharpness
                #xm.master_print("rnd_sharpness, rnd_test:", rnd_sharpness[0], rnd_test)

            if cfg.add_jpeg_artifacts:  # slow!
                inputs = (inputs.clamp(0, 1) * 255).to(torch.uint8)
                B, C, H, W = inputs.shape

                # albumentation: HWC-numpy array
                #inputs = inputs.permute(0, 2, 3, 1).reshape(B * H, W, C)
                #inputs = torch.tensor(ImageCompression(50, 100, p=1)(image=inputs.numpy())['image'])
                #inputs = inputs.reshape(B, H, W, C).permute(0, 3, 1, 2)

                # torchvision: CHW tensor
                inputs = inputs.transpose(0, 1).reshape(C, B * H, W)
                #xm.master_print("jpeg_quality:", jpeg_quality[0])
                jpeg = encode_jpeg(inputs.cpu(), jpeg_quality[0])  # batch-wise quality
                inputs = decode_jpeg(jpeg).to(device)  # jpeg tensor must be on CPU!
                inputs = inputs.reshape(C, B, H, W).transpose(1, 0)
                inputs = inputs.float() / 255

            inputs = TF.resize(inputs, cfg.size, antialias=cfg.antialias)

        # Random Noise
        if cfg.noise_level:
            # unpack labels, use same noise_level on all channels
            labels, rnd_factor = labels[:, :, :-1], labels[:, 0, -1]
            inputs += cfg.noise_level * rnd_factor[:, None, None, None] * torch.randn_like(inputs)
            inputs = inputs.clamp(0.0, 1.0)

        if cfg.curve and (cfg.curve == 'free'):
            labels = labels.reshape(labels.shape[0], -1)

        # forward
        with torch.no_grad():
            if cfg.use_aux_loss:
                seg_preds, preds = model(inputs)
            else:
                preds = model(inputs, labels) if model.requires_labels else model(inputs)
        if cfg.DEBUG and batch_idx == 1:
            xm.master_print(f"valid inputs: {inputs.shape}, value range {inputs.min():.2f} ... {inputs.max():.2f}")

        # pudae's ArcFace validation
        if cfg.pudae_valid:
            assert preds.size()[1] == 512, f'preds have wrong shape {preds.detach().size()}'
            all_scores.append(preds.detach().to(torch.float16))  # default: float32
            all_preds.append(torch.zeros_like(labels, dtype=torch.int8))
            all_labels.append(labels.to(torch.int16))  # default: int64
            continue  # skip loss

        # compute local loss
        assert preds.detach().dim() == 2, f'preds have wrong dim {preds.detach().dim()}'
        assert preds.detach().size()[1] == (cfg.n_classes or cfg.channel_size), f'preds have wrong shape {preds.detach().size()}'
        if cfg.classes:
            assert labels.max() < cfg.n_classes, f'largest label out of bound: {labels.max()}'

        if cfg.loss_weights is not None:
            if cfg.curve == 'gamma':
                _preds = [preds - torch.mean(preds, dim=1, keepdim=True), preds]
                _labels = [rel_log_gamma, abs_log_gamma]
            elif cfg.curve == 'beta':
                _preds = preds.reshape(preds.shape[0], -1, len(cfg.loss_weights)).transpose(0, 2)
                _labels = labels.reshape(labels.shape[0], -1, len(cfg.loss_weights)).transpose(0, 2)
            else:
                _preds, _labels = preds, labels

            weighted_losses = [w * criterion(p, l) for w, p, l in zip(cfg.loss_weights, _preds, _labels)]
            loss = sum(weighted_losses)
        else:
            loss = criterion(preds, labels)

        loss_meter.update(loss.item(), inputs.size(0))  # 1 aten/iter but no performance drop
        #loss_meter.update(loss.detach(), inputs.size(0))  # recursion!
        #xm.add_step_closure(loss_meter.update, args=(loss.item(), inputs.size(0)))  # recursion!
        if cfg.loss_weights is not None:
            for i, m in enumerate(weighted_loss_meters):
                m.update(weighted_losses[i].item() * cfg.n_acc, inputs.size(0))

        # torchmetrics
        if metrics:
            metrics.update(preds.detach(), labels)

    # mesh_reduce loss
    if cfg.pudae_valid:
        avg_loss = 0
    else:
        avg_loss = loss_meter.average
    if cfg.loss_weights is not None:
        xm.master_print('valid loss components:', '\t'.join([f'{m.average:.5f}' for m in weighted_loss_meters]))

    # mesh_reduce metrics
    if metrics:
        avg_metrics = metrics.compute()
        avg_metrics = {k: v.item() if v.ndim == 0 else v.tolist() for k, v in avg_metrics.items()}

        if cfg.DEBUG and 'acc' in metrics:
            counters = 'tp fp tn fn'.split()
            vals = [getattr(metrics['acc'], a).item() for a in counters]
            xm.master_print(f'metrics.acc {counters}: {vals}, sum: {sum(vals)}')

        metrics.reset()
    else:
        avg_metrics = {}

    return avg_loss, avg_metrics


def get_valid_labels(cfg, metadata):
    class_column = metadata.columns[1]  # convention, defined in metadata.get_metadata
    is_valid = metadata.is_valid
    is_shared = (metadata.fold == cfg.shared_fold) if cfg.shared_fold is not None else False
    return metadata.loc[is_valid | is_shared, class_column].values


def _mp_fn(rank, cfg, metadata, wrapped_model, xm, use_fold):
    "Distributed training loop master function"

    if cfg.xla:
        xm.master_print("In _mp_fn, world_size:", xm.xrt_world_size())

    # DDP init
    if cfg.use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # Assuming torchrun has set all env variables (LOCAL_RANK, ...).
        torch.distributed.init_process_group("nccl")

    # Agnostic device setup
    # xm.xla_device, xm.get_ordinal, rank (unused) are somewhat redundant.
    # TODO: can we merge rank==device and get rid of xm.xla_device?
    # TODO: change from xla to DDP API, adapt xla (how use xm?)
    device = xm.xla_device()
    xm.master_print("device:", device)

    # Wrap DDP model
    if cfg.use_ddp:
        wrapped_model = DDP(wrapped_model.to(device), device_ids=[device], output_device=device)
        wrapped_model.requires_labels = wrapped_model.module.requires_labels

    # XLA deviceloader
    if cfg.xla:
        import torch_xla.distributed.parallel_loader as pl
        loader_prefetch_size = 1
        device_prefetch_size = 1
        cfg.deviceloader = cfg.deviceloader or 'mp'  # 'mp' performs better than 'pl' on kaggle

    # Dataloaders
    if cfg.fake_data == 'on_device':
        is_valid = metadata.is_valid if hasattr(metadata, 'is_valid') else (metadata.fold == use_fold)
        n_valid = sum(is_valid)
        n_train = len(is_valid) - n_valid
        train_loader, valid_loader = np.arange(n_train // cfg.bs // 8), np.arange(n_valid // cfg.bs // 8)

    elif cfg.fake_data:
        train_loader, valid_loader = get_fakedata_loaders(cfg, device)

    else:
        train_loader, valid_loader = get_dataloaders(cfg, use_fold, metadata, xm)

    if hasattr(train_loader, 'sampler'):
        xm.master_print("Dataloader sampler:", train_loader.sampler.__class__.__name__)
    if hasattr(train_loader, 'num_workers'):
        xm.master_print("num_workers:", train_loader.num_workers)
    #batch = next(iter(valid_loader))  # OK
    #xm.master_print("test batch:", len(batch), batch[0].shape, batch[1].shape)

    # Send model to device
    model = wrapped_model.to(device)

    # Criterion (default reduction: 'mean'), Metrics
    criterion = nn.BCEWithLogitsLoss() if cfg.multilabel else nn.CrossEntropyLoss()
    if cfg.classes is None:
        criterion = nn.MSELoss()
    if cfg.use_aux_loss:
        from segmentation_models_pytorch.losses.dice import DiceLoss
    seg_crit = DiceLoss('binary') if cfg.use_aux_loss else None

    if cfg.loss_weights:
        cfg.loss_weights = torch.tensor(cfg.loss_weights)
    else:
        cfg.loss_weights = None

    # cfg.metrics and metrics need to have identical keys: replace aliases in cfg.metrics
    aliases = {
        'micro_acc': 'acc',
        'f1': 'F1',
        'macro_f1': 'macro_F1',
        'class_f1': 'class_F1',
        'f2': 'F2',
        'neg_rate': 'pct_N',
        'map': 'mAP',
        'eap5': 'eAP5',
        }
    cfg.metrics = cfg.metrics or []
    cfg.metrics = [k.replace(k, aliases[k]) if k in aliases else k for k in cfg.metrics]
    if cfg.save_best and cfg.save_best in aliases:
        cfg.save_best = aliases[cfg.save_best]

    # torchmetrics
    metrics = get_tm_metrics(cfg, xm)

    xm.master_print('Metrics:', metrics)

    # Don't Scale LRs (optimal lrs don't scale linearly with step size)
    lr_head, lr_bn, lr_body = cfg.lr_head, cfg.lr_bn, cfg.lr_body

    # Parameter Groups
    use_parameter_groups = False if lr_head == lr_bn == lr_body else True
    if use_parameter_groups:
        xm.master_print(f"Using parameter groups. lr_head={lr_head}, lr_body={lr_body}, lr_bn={lr_bn}")
        parameter_groups = {
            'body': (p for name, p in model.body.named_parameters() if not is_bn(name)),
            'head': model.head.parameters(),
            'bn':   (p for name, p in model.body.named_parameters() if is_bn(name)),
        }
        max_lrs = {'body': lr_body, 'head': lr_head, 'bn': lr_bn}
        params = [{'params': parameter_groups[g], 'lr': max_lrs[g]}
                  for g in parameter_groups.keys()]
        max_lrs = list(max_lrs.values())
    else:
        max_lrs = lr_head
        params = model.parameters()

    # Optimizer
    optimizer = (
        optim.AdamW(params, lr=lr_head, betas=cfg.betas, weight_decay=cfg.wd) if cfg.optimizer == 'AdamW' else
        optim.Adam(params, lr=lr_head, betas=cfg.betas)                       if cfg.optimizer == 'Adam' else
        optim.SGD(params, lr=lr_head, momentum=cfg.betas[0], dampening=1 - cfg.betas[1]))
    rst_epoch = 0
    if cfg.rst_name:
        fn = Path(cfg.rst_path) / f'{removesuffix(cfg.rst_name, ".pth")}.opt'
        if fn.exists() and not cfg.reset_opt:
            checkpoint = torch.load(fn, map_location='cpu')
            xm.master_print("Restarting from previous opt state")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            rst_epoch = checkpoint['epoch'] + 1
            cfg.optimizer_restarted = True

    # Scheduler
    if cfg.one_cycle:
        scheduler = get_one_cycle_scheduler(optimizer, max_lrs, cfg,
                                            xm, rst_epoch, train_loader)
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
        _lrs = [p["lr"] for p in optimizer.param_groups]
        xm.master_print(f"""Initial lrs: {', '.join(f'{lr:7.2e}' for lr in _lrs)}""")
        _lrs = [lr for lr in listify(max_lrs)]
        xm.master_print(f"Max lrs:     {', '.join(f'{lr:7.2e}' for lr in _lrs)}")
        if hasattr(scheduler, 'step'):
            xm.master_print(f"Batchwise stepping:", hasattr(scheduler, 'batchwise'))
    if cfg.optimizer_restarted and scheduler is None:
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = max_lrs[i] if use_parameter_groups else max_lrs

    # Maybe freeze body
    if hasattr(model, 'body') and lr_body == 0:
        xm.master_print("Freezing body of pretrained model")
        for n, p in model.body.named_parameters():
            if not is_bn(n):
                p.requires_grad = False

    model_name = f'{cfg.name}_fold{use_fold}'
    xm.master_print(f'Checkpoints will be saved as {cfg.out_dir}/{model_name}_ep*')
    step_size = cfg.bs * cfg.n_replicas * cfg.n_acc
    xm.master_print(f'Training {cfg.arch_name}, size={cfg.size}, replica_bs={cfg.bs}, '
                    f'step_size={step_size}, lr={cfg.lr_head} on fold {use_fold}')

    #
    #
    ### Training Loop ---------------------------------------------------------

    metrics_dicts = []
    best_model_score = -np.inf
    epoch_summary_header = ''.join([
        'epoch   ', ' train_loss ', ' valid_loss ',
        ' '.join([f'{key:^8}' for key in cfg.metrics if not is_listmetric(metrics[key])]),
        '   lr    ', 'min_train  min_total'])
    xm.master_print("\n", epoch_summary_header)
    xm.master_print("=" * (len(epoch_summary_header) + 2))

    for epoch in range(rst_epoch, cfg.epochs):

        # Data for verbose info
        xm.master_print(f"Epoch {epoch + 1} / {cfg.epochs}, starting at {time.strftime('%a, %d %b %Y %H:%M:%S +0000')}")
        epoch_start = time.perf_counter()
        current_lr = optimizer.param_groups[-1]["lr"]
        if hasattr(scheduler, 'get_last_lr'):
            # after restart, they disagree and scheduler is correct.
            #assert scheduler.get_last_lr()[-1] == current_lr, f'scheduler: {scheduler.get_last_lr()[-1]}, opt: {current_lr}'
            current_lr = scheduler.get_last_lr()[-1]

        # Update train_loader shuffling
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # Training Step
        if cfg.xla and (cfg.deviceloader == 'pl') and (cfg.fake_data != 'on_device'):
            # ParallelLoader requires instantiation per epoch
            dataloader = pl.ParallelLoader(train_loader, [device],
                                           loader_prefetch_size=loader_prefetch_size,
                                           device_prefetch_size=device_prefetch_size
                                           ).per_device_loader(device)
        else:
            dataloader = train_loader

        train_loss = train_fn(model, cfg, xm,
                              dataloader  = dataloader,
                              criterion   = criterion,
                              seg_crit    = seg_crit,
                              optimizer   = optimizer,
                              scheduler   = scheduler,
                              device      = device)

        # Validation Step
        valid_start = time.perf_counter()

        if cfg.train_on_all:
            valid_loss, valid_metrics = 0, []
        else:
            if cfg.xla and (cfg.deviceloader == 'pl') and (cfg.fake_data != 'on_device'):
                # ParallelLoader requires instantiation per epoch
                dataloader = pl.ParallelLoader(valid_loader, [device],
                                               loader_prefetch_size=loader_prefetch_size,
                                               device_prefetch_size=device_prefetch_size
                                               ).per_device_loader(device)
            else:
                dataloader = valid_loader

            valid_loss, valid_metrics = valid_fn(model, cfg, xm,
                                                 dataloader  = dataloader,
                                                 criterion   = criterion,
                                                 device      = device,
                                                 metrics     = metrics)

        metrics_dict = {'epoch': epoch + 1,
                        'train_loss': train_loss, 
                        'valid_loss': valid_loss}
        metrics_dict.update(valid_metrics)
        last_lr = optimizer.param_groups[-1]["lr"] if hasattr(scheduler, 'batchwise') else current_lr
        avg_lr = 0.5 * (current_lr + last_lr)
        metrics_dict['lr'] = avg_lr

        # Print epoch summary
        epoch_summary_strings = [f'{epoch + 1:>2} / {cfg.epochs:<2}']                     # ep/epochs
        epoch_summary_strings.append(f'{train_loss:10.5f}')                               # train_loss
        epoch_summary_strings.append(f'{valid_loss:10.5f}')                               # valid_loss
        for key in cfg.metrics:                                                           # metrics
            # cannot use valid_metric.items() because MetricCollection re-orders keys alphabetically
            val = valid_metrics[key]
            if isinstance(val, list):
                if getattr(metrics[key], 'average', '') is not None:
                    xm.master_print(f'Warning: metric {key} returned list but "average" attribute is not None')
                xm.master_print(key + ':\t' + "\t".join(f'{v:.5f}' for v in val))
            else:
                epoch_summary_strings.append(f'{val:7.5f}')
        epoch_summary_strings.append(f'{avg_lr:7.1e}')                                    # lr
        epoch_summary_strings.append(f'{(valid_start - epoch_start) / 60:7.2f}')          # Wall train
        epoch_summary_strings.append(f'{(time.perf_counter() - epoch_start) / 60:7.2f}')  # Wall total
        xm.master_print('  '.join(epoch_summary_strings))

        # Save weights, optimizer state, scheduler state
        # Note: xm.save must not be inside an if statement that may validate differently on
        # different TPU cores. Reason: rendezvous inside them will hang if any core
        # does not arrive at the rendezvous.
        if cfg.save_best:
            assert cfg.save_best in metrics_dict, f'{cfg.save_best} not in {list(metrics_dict)}'
        model_score = metrics_dict[cfg.save_best] if cfg.save_best else -valid_loss
        if cfg.save_best and 'loss' in cfg.save_best: model_score = -model_score
        if model_score > best_model_score or not cfg.save_best:
            if cfg.save_best:
                best_model_score = model_score
                xm.master_print(f'{cfg.save_best or "valid_loss"} improved.')
                fn = cfg.out_dir / f'{model_name}_best_{cfg.save_best}'
            else:
                fn = cfg.out_dir / f'{model_name}_ep{epoch + 1}'

            #xm.master_print(f'saving {model_name}_ep{epoch+1}.pth ...')
            xm.save((model.module if cfg.use_ddp else model).state_dict(), f'{fn}.pth')

            #xm.master_print(f'saving {model_name}_ep{epoch+1}.opt ...')
            xm.save({'optimizer_state_dict': optimizer.state_dict(),
                     'epoch': epoch}, f'{fn}.opt')

            if hasattr(scheduler, 'state_dict'):
                #xm.master_print(f'saving {model_name}_ep{epoch+1}.sched ...')
                xm.save({'scheduler_state_dict': {
                    k: v for k, v in scheduler.state_dict().items() if k != 'anneal_func'}},
                    f'{fn}.sched')

        # Save metrics in pth and csv file (rank 0 only)
        metrics_dict['Wall'] = (time.perf_counter() - epoch_start) / 60

        metrics_dicts.append(metrics_dict)
        xm.save({k: [d[k] for d in metrics_dicts] for k in metrics_dicts[0]}, 
                Path(cfg.out_dir) / f'metrics_fold{use_fold}.pth')

        if (cfg.n_replicas == 1) or (xm.get_ordinal() == 0):
            csv_file = Path(cfg.out_dir) / f'metrics_fold{use_fold}.csv'
            if csv_file.exists():
                with open(csv_file, 'r') as fp:
                    keys = fp.readline().strip().split(',')
            else:
                keys = list(metrics_dict.keys())
                with open(csv_file, 'w') as fp:
                    fp.write(','.join(keys) + '\n')
            line_str = ','.join(str(metrics_dict[key]) for key in keys)
            with open(csv_file, 'a') as fp:
                fp.write(line_str + '\n')

    if cfg.use_ddp:
        torch.distributed.destroy_process_group()
