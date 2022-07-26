import numpy as np
import torch
from torch.optim import lr_scheduler


def maybe_step(scheduler, xm):
    """Step scheduler until it reaches its total_steps"""
    if getattr(scheduler, 'last_epoch', -1) < getattr(scheduler, 'total_steps', np.inf):
        scheduler.step()
    else:
        xm.master_print("WARNING: scheduler reached total_steps of", scheduler.total_steps)


def get_one_cycle_scheduler(optimizer, max_lr, cfg, xm, rst_epoch=0, dataloader=None):

    # Get total_steps and last_steps
    frac = cfg.frac if (cfg.do_class_sampling or cfg.filetype == 'tfrec') else 1
    num_cores = cfg.n_replicas if cfg.xla else 1
    n_examples = cfg.NUM_TRAINING_IMAGES
    steps_per_epoch = int(n_examples * frac) // (cfg.bs * num_cores * cfg.n_acc)
    # steps_per_epoch can be short by 1 even if drop_last==True: happywhale_classifier_deeptrane2 v62
    #xm.master_print("n_examples:", n_examples, "frac:", frac)
    #xm.master_print("cfg.bs:", cfg.bs, "num_cores:", num_cores, "n_acc:", n_acc)
    #xm.master_print("denominator:", (cfg.bs * num_cores * n_acc))
    total_steps = (rst_epoch + cfg.epochs) * steps_per_epoch
    last_step = rst_epoch * steps_per_epoch - 1 if rst_epoch else -1

    xm.master_print(f"Steps per epoch: {steps_per_epoch}")
    xm.master_print(f"Total steps:     {total_steps} (my guess)")

    # Alternative total_steps from dataloader (map-style datasets only,
    # sequential loaders/iterable-style datasets have __len__ that raises TypeError!)
    try:
        n_batches = len(dataloader)
        total_steps = (rst_epoch + cfg.epochs) * (n_batches // cfg.n_acc)
        xm.master_print(f"Total steps:     {total_steps} (dataloader)")
    except (TypeError, AttributeError):
        pass

    if rst_epoch != 0:
        xm.master_print(f"Restart epoch: {rst_epoch}")
        xm.master_print(f"Last step: {last_step}")

    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps=total_steps,
                                        last_epoch=last_step,
                                        div_factor=cfg.div_factor, pct_start=cfg.pct_start)

    # Warm restart
    fn = Path(cfg.rst_path or '.') / f'{removesuffix(cfg.rst_name or "", ".pth")}.sched'
    if fn.exists() and not cfg.reset_opt:
        checkpoint = torch.load(fn, map_location='cpu')
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        xm.master_print("Restarting from previous scheduler state")

    scheduler.batchwise = True

    return scheduler