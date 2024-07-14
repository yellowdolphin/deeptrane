import os
import sys
import importlib
from pathlib import Path
from time import perf_counter
from multiprocessing import cpu_count

from config import Config, parser
from utils.general import quietly_run, sizify, listify, autotype, get_drive_out_dir
from utils.tf_setup import install_model_libs

# Read config file and parser_args
parser_args, _ = parser.parse_known_args(sys.argv)
print("[ √ ] Config file:", parser_args.config_file)
cfg = Config('configs/defaults')
if parser_args.config_file: cfg.update(parser_args.config_file)

cfg.mode = parser_args.mode
cfg.use_folds = parser_args.use_folds or cfg.use_folds
cfg.epochs = parser_args.epochs or cfg.epochs
cfg.rst_epoch = (cfg.rst_epoch or 0) if (cfg.rst_path and cfg.rst_name) else 0
cfg.size = cfg.size if parser_args.size is None else sizify(parser_args.size)
cfg.betas = parser_args.betas or cfg.betas
for key in 'dropout_ps lin_ftrs freeze_for_loading freeze'.split():
    setattr(cfg, key, cfg[key] if getattr(parser_args, key) is None else listify(getattr(parser_args, key)))
cfg.batch_verbose = cfg.batch_verbose or 'auto'
for key, value in listify(parser_args.set):
    autotype(cfg, key, value)
if cfg.batch_verbose not in ['auto', 0, 1, 2]:
    print(f'WARNING: batch_verbose {cfg.batch_verbose} ignored.')
    print('Valid values are 0 = silent, 1 = progress bar, 2 = one line per epoch.')
cfg.cloud = 'drive' if os.path.exists('/content') else 'kaggle' if os.path.exists('/kaggle') else 'gcp'
if cfg.cloud == 'drive':
    cfg.out_dir = get_drive_out_dir(cfg)  # config.yaml and experiments go there

print(cfg)
print("[ √ ] Cloud:", cfg.cloud)
print("[ √ ] Tags:", cfg.tags)
print("[ √ ] Mode:", cfg.mode)
print("[ √ ] Folds:", cfg.use_folds)
print("[ √ ] Architecture:", cfg.arch_name)

cfg.save_yaml()

# Config consistency checks
if cfg.rst_name is not None:
    rst_file = Path(cfg.rst_path) / cfg.rst_name
    assert rst_file.exists(), f'{rst_file} not found'  # fail early
cfg.out_dir = Path(cfg.out_dir)

# Suppress warnings/errors (no cuda on TPU machines, etc.)
if not cfg.DEBUG:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Installs and imports
install_model_libs(cfg)
import torch  # temporary: import prior to tf to work around 2.0 import issue
print("[ √ ] torch:", torch.__version__)
import tensorflow as tf
print("[ √ ] tf:", tf.__version__)
import tf_keras
print("[ √ ] tf_keras:", tf_keras.__version__)
from tf_keras.backend import clear_session  # no tf_keras? see install_model_libs!
if cfg.cloud == 'kaggle' and cfg.normalize in ['torch', 'tf', 'caffe']:
    quietly_run(f'pip install keras=={tf_keras.__version__}')
from utils.tensorflow import get_lr_callback, CSVLogger
from models_tf import get_pretrained_model
from tf_data import cv_split, split_by_name, count_data_items, get_dataset, configure_data_pipeline

# Import project (code, constant settings)
project = importlib.import_module(f'projects.{cfg.project}') if cfg.project else None
if project:
    print("[ √ ] Project:", cfg.project)
    project.init(cfg)

# TPU detection
print("tf.config.list_physical_devices():", tf.config.list_physical_devices())
print("tf.config.list_physical_devices('TPU'):", tf.config.list_physical_devices('TPU'))
print("TPU_NAME:", os.environ.get('TPU_NAME', None))
if tf.config.list_physical_devices('TPU') or ('TPU_NAME' in os.environ):
    print("TPUs detected, trying to use them...")
    try:
        if cfg.cloud in ['kaggle', 'drive']:
            # TPUClusterResolver.connect() on kaggle raises ValueError: temporary google issue
            # initialize_tpu_system(tpu) on kaggle raises "No matching devices found":
             #    -> make sure, tensorflow is not imported in the notebook before running deeptrane!
            # TPUClusterResolver.connect() on colab sometimes does not respond
            if 'TPU_NAME' in os.environ:
                # No parameters necessary if TPU_NAME environment variable is set.
                # This is always the case on Kaggle.
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            else:
                print("trying TPUClusterResolver(tpu='local')...")
                tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
        else:
            print("trying TPUClusterResolver.connect()...")
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

        print('TPU master:', tpu.master())
        strategy = tf.distribute.TPUStrategy(tpu)

        # Workaround issue https://github.com/tensorflow/hub/issues/604
        os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"
    except Exception as e:
        print(e)
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        print("Fallback to CPU/GPU strategy")
        strategy = tf.distribute.get_strategy()  # default for cpu, gpu
else:
    strategy = tf.distribute.get_strategy()  # default for cpu, gpu

print(f"[ √ ] {cpu_count()} CPUs")
cfg.n_replicas = strategy.num_replicas_in_sync
cfg.gpu = len(tf.config.list_physical_devices('GPU'))
if cfg.gpu:
    print(f"[ √ ] {cfg.gpu} GPU(s) found")
elif cfg.n_replicas > 1:
    print("[ √ ] Number of replicas:", cfg.n_replicas)
else:
    cfg.bs = min(cfg.bs, 3 * cpu_count())  # avoid RAM exhaustion during CPU debug
    print(f"[ √ ] No accelerators found, reducing bs to {cfg.bs}")

configure_data_pipeline(cfg)


# CV Training
for use_fold in cfg.use_folds:

    if hasattr(project, 'split'):
        project.split(cfg)
    elif cfg.split_by_name:
        split_by_name(cfg)
    else:
        cv_split(cfg, use_fold)
    assert (cfg.train_files is not None) and (cfg.valid_files is not None)

    # configure number of examples for training and validation
    if hasattr(project, 'count_data_items'):
        cfg.num_train_images = project.count_data_items(cfg.train_files, cfg.tfrec_filename_pattern)
        cfg.num_valid_images = project.count_data_items(cfg.valid_files, cfg.tfrec_filename_pattern)
    else:
        cfg.num_train_images = count_data_items(cfg.train_files, cfg.tfrec_filename_pattern)
        cfg.num_valid_images = count_data_items(cfg.valid_files, cfg.tfrec_filename_pattern)
    if cfg.frac is not None:
        frac_train, frac_valid = (cfg.frac, cfg.frac) if isinstance(cfg.frac, (int, float)) else cfg.frac
        cfg.num_train_images = int(frac_train * cfg.num_train_images)
        cfg.num_valid_images = int(frac_valid * cfg.num_valid_images)
    print(f"         urls    examples")
    print(f"train: {len(cfg.train_files):>6}  {cfg.num_train_images:>10}")
    print(f"valid: {len(cfg.valid_files):>6}  {cfg.num_valid_images:>10}")

    train_dataset = get_dataset(cfg, project, mode='train')
    valid_dataset = get_dataset(cfg, project, mode='valid')

    dataloader_bs = cfg.bs * cfg.n_replicas
    batches_per_epoch = cfg.num_train_images // dataloader_bs
    valid_batches_per_epoch = cfg.num_valid_images // dataloader_bs
    steps_per_epoch = batches_per_epoch // cfg.n_acc   ## check: float possible? is last opt step skipped?
    valid_steps_per_epoch = valid_batches_per_epoch // cfg.n_acc
    cfg.validation_steps = cfg.validation_steps or valid_steps_per_epoch
    step_size = cfg.bs * cfg.n_replicas * cfg.n_acc
    if hasattr(train_dataset, 'batch_size'): 
        print("dataset.batch_size:", train_dataset.batch_size)
    print("dataloader_bs:", dataloader_bs)
    print("batches_per_epoch:", batches_per_epoch)
    print("validation_steps:", cfg.validation_steps)

    # Training callbacks
    logfile = f'{cfg.out_dir}/metrics_fold{use_fold}.csv'
    csv_logger = CSVLogger(logfile)
    cfg.checkpoint_format = cfg.checkpoint_format or 'hdf5'
    chk_ep = '_best' if cfg.save_best else '_ep{epoch:03d}'
    chk_suffix = '.keras' if (cfg.checkpoint_format == 'keras') else '.h5'
    # TF < 2.16 bug: saves entire model if suffix is ".weights.h5" but TF 2.16 asserts suffix is ".weights.h5"
    # tf_keras < 2.16.0 bug: saves entire model if suffix is ".weights.h5", will tf_keras 2.16.1 fix it?
    if int(tf_keras.__version__.split('.')[1]) > 16 and not cfg.save_full_model:
        chk_suffix = '.weights.h5'
    chk_filepath = f'{cfg.out_dir}/{cfg.arch_name}{chk_ep}{chk_suffix}'
    
    save_chk = (
        tf_keras.callbacks.ModelCheckpoint(
            chk_filepath, save_best_only=True,
            monitor=f'val_{"arc_" if (cfg.arcface and cfg.aux_loss) else ""}{cfg.save_best}', 
            mode='min' if 'loss' in cfg.save_best else 'max',
            save_weights_only=(not cfg.save_full_model),
            save_freq='epoch', verbose=1) if cfg.save_best else
        tf_keras.callbacks.ModelCheckpoint(
            chk_filepath, save_best_only=False,
            save_weights_only=(not cfg.save_full_model),
            save_freq='epoch', verbose=1))
    print(f"save_weights_only: {save_chk.save_weights_only}")
    lr_callback = get_lr_callback(cfg)

    clear_session()

    if hasattr(project, 'get_pretrained_model'):
        model = project.get_pretrained_model(cfg, strategy)
    else:
        model = get_pretrained_model(cfg, strategy)
    model.summary(line_length=120)
    ### DEBUG
    #for w in model.layers[2].weights:
    #    print(w.name, w.shape)

    t0 = perf_counter()

    print(f'\nTraining {cfg.arch_name}, size={cfg.size}, replica_bs={cfg.bs}, '
          f'step_size={step_size}, lr={cfg.lr} on fold {use_fold}')
    cfg.rst_epoch = cfg.rst_epoch or 0

    history = model.fit(train_dataset,
                        validation_data=valid_dataset if cfg.validation_steps else None,
                        steps_per_epoch=batches_per_epoch,
                        validation_steps=cfg.validation_steps,
                        epochs=cfg.epochs,
                        callbacks=[lr_callback, save_chk, csv_logger],
                        verbose=cfg.batch_verbose,
                        initial_epoch=cfg.rst_epoch,
                        )
    wall = perf_counter() - t0
    min_per_ep = wall / (cfg.epochs - cfg.rst_epoch or 1) / 60
    print(f"Training finished in {wall / 60:.2f} min ({min_per_ep:.2f} min/epoch)")

    if hasattr(project, 'on_train_end'):
        project.on_train_end(cfg, model, history)

    # avoid Exception in function AtomicFunction.__del__ (TF bug since 2.13.0)
    del train_dataset, valid_dataset
