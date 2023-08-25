# Setup
project = 'autolevels'
datasets = ['imagenet-1k-tfrecords-ilsvrc2012-part-0', 
            'imagenet-1k-tfrecords-ilsvrc2012-part-1']
gcs_paths = ['gs://kds-427f890caad365e12bf0dd8711053e1866927a0ecc5e9dad92c683e5', 
             'gs://kds-667913c26db265c8bf8f854d15cf726d94bb8992aec4a1eb8e48f5df']
gcs_filters = ['*/*-of-*', '*-of-*']
tfrec_filename_pattern = r"-of-([0-9]*)$"
out_dir = '/kaggle/working'
use_custom_training_loop = False

# Training
seed = 42
num_folds = 5
use_folds = [0]
train_on_all = False
size = (384, 384)
multilabel = False
metrics = ['curve_rmse']
no_macro_metrics = True  # otherwise slow valid + 8-TPU-issue
bs = 64
steps_per_execution = 1  # increase for performance (check callbacks, training behavior)
epochs = 4
batch_verbose = 1
lr = 1e-3
one_cycle = True
div_factor = 5                 # default: 25, from Chest14: 1
pct_start = 0.25               # default: 0.3, from Chest14: 0.6, pipeline1: 0
lr_min = 1e-6
save_best = 'loss'
predict_inverse = True
output_curve_params = False
blackpoint_range = (-30, 30)   # x-offset
blackpoint2_range = (-30, 30)  # y-offset
log_gamma_range = [-1.8, 1.8]
add_uniform_noise = True       # add uniform noise to mask uint8 discretization [bool|float]
add_jpeg_artifacts = False
sharpness_augment = False
noise_level = 0.01             # random normal noise (augmentation)
augmentation = 'autolevels_aug_tf'

# Model
arch_name = 'efnv2s'
bn_eps = 1e-5
#rst_path = '/kaggle/input/cassava-deeptrane-rst'
#rst_name = ''
optimizer = "Adam"  # Adam AdamW SGD
dropout_ps = []


from pathlib import Path

_accepted_types = (int, float, str, bool, tuple, list, dict)
cfg = {k: v for k, v in globals().items() if not k.startswith('_') and isinstance(v, _accepted_types)}

cfg["name"] = Path(__file__).stem
cfg["tags"] = cfg["name"].split("_")

# project-dependent settings (deprecated, put in project module)
cfg["curve"] = 'gamma' if 'gamma' in cfg["tags"] else 'beta'
cfg["pretrained"] = 'noisy-student' if 'v1' in cfg["arch_name"] else 'imagenet21k-ft1k'
if cfg["epochs"] == 1: cfg["batch_verbose"] = 1
