"""
The Issue with learning-rate schedulers in TF

tf.keras.model.fit() allows two ways to schedule the lr:
    (1) pass a LearningRateSchedule as learning_rate to optimizer
    (2) pass a LearningRateScheduler as callback to fit()

(1) May update the lr at each optimizer step (good) but CSVLogger does not log the lr (bad). 
Could not confirm correct lr on TPUs.

(2) Updates the lr once per epoch (bad), CSVLogger logs the lr (good), tracing may occur (bad).
Do not pass `steps_per_epoch` when choosing this option!
"""

import math
from time import sleep
import matplotlib.pyplot as plt
import tensorflow as tf


def get_lr_callback(cfg, steps_per_epoch=1, plot=False):
    assert cfg.lr is not None, 'config lacks lr'
    assert cfg.lr_min is not None, 'config lacks lr_min'
    assert cfg.pct_start is not None, 'config lacks pct_start'
    assert cfg.epochs is not None, 'config lacks epochs'
    decay = cfg.lr_decay or 'cos'
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

    #@tf.function
    def lrfn(iterations):
        # iterations can be tensor with dtype=tf.float32.
        # When wrapped in LearningRateScheduler and called from model.fit(callbacks=lr_callback), iterations is
        # int and offset by cfg.rst_epoch if initial_epoch=cfg.rst_epoch is passed to fit() as well.
        #print("lrfn called with", float(iterations), f"({iterations.dtype if hasattr(iterations, 'dtype') else type(iterations)})")
        #print("rst_epoch:", rst_epoch)
        # lrfn will be called at epoch_start.
        # If passed inside LRSchedule to optimizer, it may be called with optimizer.iterations, once per optimizer step.

        # WARNING:tensorflow:5 out of the last 5 calls to <function get_lr_callback.<locals>.lrfn at 0x7f9e045e43b0> triggered 
        # tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to 
        # (1) creating @tf.function repeatedly in a loop, [NO]
        # (2) passing tensors with different shapes, [NO]
        # (3) passing Python objects instead of tensors. [NO]
        # For (1), please define your @tf.function outside of the loop. For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.
        # - tf2.4.1@kaggle no warn w ramp/lr_max (no elif)
        # - tf2.4.1@kaggle no warn w ramp/cos (no elif)
        # - tf2.8.2@colab warns in ep 4, 5 w ramp/lr_sus_ep/cos (1 elif)
        # - tf2.8.2@colab warns in ep 4, 5 w ramp/cos (no elif)
        # - tf2.4.1@kaggle no warn w ramp/lr_sus_ep/cos (1 elif)
        # - tf2.8.2@colab warns in ep 4, w return max_lr (max_lr is tf.constant in closure)
        # - tf2.8.2@colab warns w return tf.constant(3e-4, dtype=tf.float32) (no closure)
        # - tf2.8.2@colab no warn w all but w/o @tf.function
        # => tf2.4.1 does not warn at all and tf2.8.2 retraces any lrfn if decorated with tf.function.
        # Effect on performance is minor if any.

        epoch = (iterations / steps_per_epoch if type(iterations) is int else
                 iterations / steps_per_epoch + rst_epoch)

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

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=cfg.DEBUG)
    # verbose prints lr at epoch start

    return lr_callback


class LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, sched, **kwargs):
        super().__init__()
        is_callback = isinstance(sched, tf.keras.callbacks.LearningRateScheduler)
        self.lrfn = sched.schedule if is_callback else sched

    def __call__(self, step):
        return self.lrfn(step)


class CSVLogger(tf.keras.callbacks.CSVLogger):
    """
    Closes self.csv_file after each epoch.
    Otherwise, on google drive, csv_file is not created when training is interrupted.

    Increments epoch in csv file to match keras.model.fit output and checkpoint names.
    """
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch + 1, logs=logs)
        #self.csv_file.close()
        self.on_train_end(logs=logs)
        self.append = True

        # wait till file is created before calling on_train_begin
        seconds_waited, timeout = 0, 60
        while not tf.io.gfile.exists(self.filename):
            tf.print(f"waiting for {self.filename}")
            seconds_waited += 5
            sleep(5)
            if seconds_waited > timeout:
                new_filename = self.filename.split('_ep')[0].replace('.csv', '') + f'_ep{epoch + 1}.csv'
                tf.print(f"CSVLogger timeout: {self.filename} not created, changing to {new_filename}")
                self.filename = new_filename
                self.append = False
                break
        self.on_train_begin(logs=logs)
