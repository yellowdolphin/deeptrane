# Setup
project = 'autolevels'
datasets = [f'google-landmark-tfrecords-512-{i}' for i in range(1, 7)]
gcs_paths = ['gs://kds-8285bf4361c1ca055187f1d44d73d2f621d0c5c680d39013e79c970f', 'gs://kds-c1e675145dd782b54b1d2695e1ab842acd7618ae39225df25aaa5883', 'gs://kds-8b820e025265b89fff9341fde7e3206d40d7f4bd5ae902a53febe7fb', 'gs://kds-619c3bf64c1022adbdd2ce0f41171fc8221827f13346c3aaf6244d98', 'gs://kds-09fa8bd5d5990b17e83f006931d807ec78f37d6e4f1077a68e042149', 'gs://kds-9a6ae5ae37644a36de4648ca99a776ab95e20e2d9d3723c9bfb818e2']
gcs_filter = '*.tfrec'
tfrec_filename_pattern = None
BGR = True
out_dir = '/kaggle/working'
use_custom_training_loop = False

# Training
seed = 42
num_folds = 10
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
div_factor = 5                            # default: 25, from Chest14: 1
pct_start = 0.25                           # default: 0.3, from Chest14: 0.6, pipeline1: 0
lr_min = 1e-6
save_best = 'loss'
preprocess = 'curve4'
freeze_for_loading = ['none']
freeze = ['body', 'head']
predict_inverse = True
blackpoint_range = (0, 0)   # x-offset
blackpoint2_range = (0, 0)  # y-offset
log_gamma_range = [0, 0]
mirror_gamma = False
p_gamma = 1.0                  # probability for using Gamma curve
p_beta = 0.0
curve3_a_range = (1, 1) #(0.2, 1.06)(0.3, 0.693) #(0.34, 1.06)  # log_alpha
curve3_beta_range = (1, 1)
curve4_loga_range = (0, 0)
curve4_b_range = (1, 1)
add_uniform_noise = False      # add uniform noise to mask uint8 discretization [bool|float]
add_jpeg_artifacts = False
sharpness_augment = False
noise_level = 0.0              # random normal noise (augmentation)
augmentation = 'autolevels_aug_tf'

# Model
arch_name = 'efnv2s'
bn_eps = 1e-5
#rst_path = '/kaggle/input/cassava-deeptrane-rst'
#rst_name = ''
optimizer = "Adam"  # Adam AdamW SGD
wd = 5e-2
dropout_ps = [0, 0, 0, 0]
lin_ftrs = [24, 768, 768, 768]
act_head = 'silu'


from pathlib import Path

_accepted_types = (int, float, str, bool, tuple, list)
cfg = {k: v for k, v in globals().items() if not k.startswith('_') and isinstance(v, _accepted_types)}

cfg["name"] = Path(__file__).stem
cfg["tags"] = cfg["name"].split("_")

# project-dependent settings (deprecated, put in project module)
cfg["curve"] = 'gamma' if 'gamma' in cfg["tags"] else 'beta' if 'beta' in cfg["tags"] else 'free'
cfg["pretrained"] = 'noisy-student' if 'v1' in cfg["arch_name"] else 'imagenet21k-ft1k'
if cfg["epochs"] == 1: cfg["batch_verbose"] = 1
