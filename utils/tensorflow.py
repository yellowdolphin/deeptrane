"""
The Issue with learning-rate schedulers in TF

tf.keras.model.fit() allows two ways to schedule the lr, (1) pass a 
LearningRateSchedule as learning_rate to optimizer, or (2) pass a
LearningRateScheduler as callback to fit().

(1) May update the lr at each optimizer step (good) but CSVLogger does not
log the lr (bad). Could not confirm correct lr on TPUs.

(2) Updates the lr once per epoch (bad), CSVLogger logs the lr (good), tracing may occur (bad).
    => Do not pass `steps_per_epoch`!
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
    rst_epoch  = tf.constant(cfg.rst_epoch, dtype=tf.float32)
    pi         = tf.constant(math.pi, dtype=tf.float32)

    @tf.function
    def lrfn(iterations):
        # iterations is tensor with dtype=tf.float32
        #tf.print("lrfn called with", float(iterations), f"({iterations.dtype if hasattr(iterations, 'dtype') else type(iterations)})")
        # lrfn will be called at epoch_start.
        # If passed inside LRSchedule to optimizer, it may be called with optimizer.iterations, once per optimizer step.
        epoch = iterations / steps_per_epoch + rst_epoch

        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start  # tracing Warning in ep 4, 5
# WARNING:tensorflow:5 out of the last 5 calls to <function get_lr_callback.<locals>.lrfn at 0x7f9e045e43b0> triggered 
# tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to 
# (1) creating @tf.function repeatedly in a loop, [NO]
# (2) passing tensors with different shapes, [NO]
# (3) passing Python objects instead of tensors. [NO]
# For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.            

        #elif epoch < lr_ramp_ep + lr_sus_ep:
        #    lr = lr_max

        #elif decay == 'exp':
        #    lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min

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
