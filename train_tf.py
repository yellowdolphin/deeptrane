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
cfg.rst_epoch = cfg.rst_epoch or 0
cfg.size = cfg.size if parser_args.size is None else sizify(parser_args.size)
cfg.betas = parser_args.betas or cfg.betas
cfg.dropout_ps = cfg.dropout_ps if parser_args.dropout_ps is None else listify(parser_args.dropout_ps)
cfg.lin_ftrs = cfg.lin_ftrs if parser_args.lin_ftrs is None else listify(parser_args.lin_ftrs)
cfg.batch_verbose = cfg.batch_verbose or 'auto'
for key, value in listify(parser_args.set):
    autotype(cfg, key, value)
if cfg.batch_verbose not in ['auto', 0, 1, 2]:
    print(f'WARNING: batch_verbose {cfg.batch_verbose} ignored.')
    print('Valid values are 0 = silent, 1 = progress bar, 2 = one line per epoch.')
print(cfg)

cfg.cloud = 'drive' if os.path.exists('/content') else 'kaggle' if os.path.exists('/kaggle') else 'gcp'
if cfg.cloud == 'drive':
    cfg.out_dir = get_drive_out_dir(cfg)  # config.yaml and experiments go there
print("[ √ ] Cloud:", cfg.cloud)
print("[ √ ] Tags:", cfg.tags)
print("[ √ ] Mode:", cfg.mode)
print("[ √ ] Folds:", cfg.use_folds)
print("[ √ ] Architecture:", cfg.arch_name)

cfg.save_yaml()

# Config consistency checks
if cfg.rst_name is not None:
    rst_file = Path(cfg.rst_path) / f'{cfg.rst_name}.pth'
    assert rst_file.exists(), f'{rst_file} not found'  # fail early
cfg.out_dir = Path(cfg.out_dir)

# Suppress warnings/errors (no cuda on TPU machines, etc.)
if not cfg.DEBUG:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Installs and imports
install_model_libs(cfg)
import tensorflow as tf
print("[ √ ] tf:", tf.__version__)
if cfg.cloud == 'kaggle' and cfg.normalize in ['torch', 'tf', 'caffe']:
    quietly_run(f'pip install keras=={tf.keras.__version__}')
from tensorflow.keras.backend import clear_session
from models_tf import get_pretrained_model
import tf_data
from tf_data import (get_gcs_path, cv_split, count_data_items,
    get_dataset, configure_data_pipeline)
from utils.tensorflow import get_lr_callback

# Import project (code, constant settings)
project = importlib.import_module(f'projects.{cfg.project}') if cfg.project else None
if project:
    print("[ √ ] Project:", cfg.project)
    project.init(cfg)


# TPU detection. No parameters necessary if TPU_NAME environment variable is set.
# This is always the case on Kaggle.
try:
    #tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    #tf.config.experimental_connect_to_cluster(tpu)
    #tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    
    # Workaround issue https://github.com/tensorflow/hub/issues/604
    os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"
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

cfg.gcs_path = cfg.gcs_path or get_gcs_path(cfg, project)

configure_data_pipeline(cfg, project)


# CV Training
for use_fold in cfg.use_folds:

    if hasattr(project, 'split'):
        project.split(cfg)
    else:
        cv_split(cfg, use_fold)
    assert cfg.train_files and cfg.valid_files

    if hasattr(project, 'count_examples'):
        cfg.num_train_images = project.count_data_items(cfg.train_files)
        cfg.num_valid_images = project.count_data_items(cfg.valid_files)
    else:
        cfg.num_train_images = count_data_items(cfg.train_files)
        cfg.num_valid_images = count_data_items(cfg.valid_files)
    print(f"         urls    examples")
    print(f"train: {len(cfg.train_files):>6}  {cfg.num_train_images:>10}")
    print(f"valid: {len(cfg.valid_files):>6}  {cfg.num_valid_images:>10}")

    train_dataset = get_dataset(cfg, mode='train')
    valid_dataset = get_dataset(cfg, mode='valid')

    dataloader_bs = cfg.bs * cfg.n_replicas
    batches_per_epoch = cfg.num_train_images // dataloader_bs
    steps_per_epoch = batches_per_epoch // cfg.n_acc   ## check: float possible? is last opt step skipped?
    step_size = cfg.bs * cfg.n_replicas * cfg.n_acc
    if hasattr(train_dataset, 'batch_size'): tf.print("dataset.batch_size:", train_dataset.batch_size)
    tf.print("dataloader_bs:", dataloader_bs)
    tf.print("batches_per_epoch:", batches_per_epoch)

    # Training callbacks
    train_logger = tf.keras.callbacks.CSVLogger(f'{cfg.out_dir}/metrics_fold{use_fold}.csv')
    save_best = tf.keras.callbacks.ModelCheckpoint(
        f'{cfg.out_dir}/{cfg.arch_name}_best.h5', save_best_only=True,
        monitor=f'val_{"arc_" if (cfg.arcface and cfg.aux_loss) else ""}{cfg.save_best}', 
        mode='min' if 'loss' in cfg.save_best else 'max',
        save_weights_only=True, save_freq='epoch', verbose=1)
    lr_callback = get_lr_callback(cfg)

    clear_session()

    if hasattr(project, 'get_pretrained_model'):
        model = project.get_pretrained_model(cfg, strategy)
    else:
        model = get_pretrained_model(cfg, strategy)

    if cfg.rst_path and cfg.rst_name:
        model.load_weights(Path(cfg.rst_path) / cfg.rst_name)
        print(f"Weights loaded from {cfg.rst_name}")

    t0 = perf_counter()

    print(f'Training {cfg.arch_name}, size={cfg.size}, replica_bs={cfg.bs}, '
          f'step_size={step_size}, lr={cfg.lr} on fold {use_fold}')
    cfg.rst_epoch = cfg.rst_epoch or 0

    history = model.fit(train_dataset,
                        validation_data=valid_dataset,
                        steps_per_epoch=batches_per_epoch,
                        epochs=cfg.epochs - cfg.rst_epoch,
                        callbacks=[train_logger, save_best, lr_callback],
                        verbose=cfg.batch_verbose,
                        initial_epoch=cfg.rst_epoch,
                        )
    wall = perf_counter() - t0
    min_per_ep = wall / cfg.epochs / 60
    print(f"Training finished in {wall / 60:.2f} min ({min_per_ep:.2f} min/epoch)")
