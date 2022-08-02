"""
The Issue with learning-rate schedulers in TF

tf.keras.model.fit() allows two ways to schedule the lr, (1) pass a 
LearningRateSchedule as learning_rate to optimizer, or (2) pass a
LearningRateScheduler as callback to fit().

(1) updates the lr at each optimizer step (good) but CSVLogger does not
log the lr (bad).

(2) updates the lr once per epoch (bad), CSVLogger logs the lr (good).

How log lr in case of (1)?
   - add lr as metric?
   - does History callback return lrs? Probably not, CSVLogger might log all that History returns
"""

import math
import matplotlib.pyplot as plt
import tensorflow as tf


def get_lr_callback(cfg, decay='cos', steps_per_epoch=1, plot=False):
    assert cfg.lr, 'config lacks lr'
    assert cfg.lr_min, 'config lacks lr_min'
    assert cfg.pct_start, 'config lacks pct_start'
    assert cfg.epochs, 'config lacks epochs'
    steps_per_epoch = tf.constant(steps_per_epoch, dtype=tf.float32)
    lr_max     = tf.constant(cfg.lr, dtype=tf.float32)
    lr_start   = tf.constant(0.2 * lr_max, dtype=tf.float32)
    lr_min     = tf.constant(cfg.lr_min * lr_max, dtype=tf.float32)
    lr_ramp_ep = tf.constant(cfg.pct_start * cfg.epochs, dtype=tf.float32)
    lr_sus_ep  = tf.constant(0, dtype=tf.float32)
    lr_decay   = tf.constant(0.9, dtype=tf.float32)
    n_epochs   = tf.constant(cfg.epochs, dtype=tf.float32)
    restart    = cfg.rst_path and cfg.rst_name
    rst_epoch  = tf.constant(cfg.rst_epoch or 0, dtype=tf.float32)
    pi         = tf.constant(math.pi, dtype=tf.float32)

    @tf.function
    def lrfn(iterations):
        # iterations is tensor with dtype=tf.float32
        tf.print("lrfn called with", float(iterations), f"({iterations.dtype if hasattr(iterations, 'dtype') else type(iterations)})")
        # Issue: If model is built with optimizer with constant lr, 
        # learning_rate will be called with epoch rather than optimizer.iterations
        epoch = iterations / steps_per_epoch
        if restart:
            epoch += cfg.rst_epoch
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        elif decay == 'exp':
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
        else:
            x = (epoch - lr_ramp_ep - lr_sus_ep) / (n_epochs - lr_ramp_ep - lr_sus_ep)
            lr = (lr_min + lr_max) / 2 + (lr_max - lr_min) / 2 * tf.math.cos(pi * x)
            
        return lr
        
    if plot:
        n_ep = cfg.epochs - cfg.rst_epoch if (cfg.rst_path and cfg.rst_name) else cfg.epochs
        epochs = tf.range(n_ep, dtype=tf.float32)
        learning_rates = [lrfn(e).numpy() for e in epochs]
        plt.plot(epochs.numpy(), learning_rates)
        plt.show()

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)

    return lr_callback


class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, sched, **kwargs):
        super().__init__()
        is_callback = isinstance(sched, tf.keras.callbacks.LearningRateScheduler)
        self.lrfn = sched.schedule if is_callback else sched

    def __call__(self, step):
        return self.lrfn(step)
