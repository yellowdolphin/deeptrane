# Setup
project = 'autolevels'
datasets = ['imagenet-1k-tfrecords-ilsvrc2012-part-0', 
            'imagenet-1k-tfrecords-ilsvrc2012-part-1']
gcs_paths = ['gs://kds-84ef4b3d7d4d649d5d74b74127d9161471546611b520314672040df0', 'gs://kds-1a0e526d26959908322c5ed6f78de83de9ae67a476dc20f7f15d8329']
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
div_factor = 5                            # default: 25, from Chest14: 1
pct_start = 0.25                           # default: 0.3, from Chest14: 0.6, pipeline1: 0
lr_min = 1e-6
save_best = 'loss'
predict_inverse = True
blackpoint_range = (-140, 140)   # x-offset
blackpoint2_range = (-140, 140)  # y-offset
log_gamma_range = [-1.8, 1.8]
curve3_a_range = (0.2, 1.06) #(0.2, 1.06)(0.3, 0.693) #(0.34, 1.06)  # log_alpha
curve3_beta_range = (0.5, 1)
mirror_beta = False
curve4_loga_range = (-1.8, 0.0)
curve4_b_range = (0.4, 1.2)
mirror_curve4 = True
p_gamma = 0.4                  # probability for using Gamma curve
p_beta = 0.33                  # probability for using Beta PDF rather than Curve4
add_uniform_noise = False      # add uniform noise to mask uint8 discretization [bool|float]
add_jpeg_artifacts = True
sharpness_augment = True
noise_level = 0.01             # random normal noise (augmentation)
resize_before_jpeg = True      # resize already in curve_tfm_image
augmentation = 'autolevels_aug_tf'
catmix = False                 # compose input of 4 images

# Model
arch_name = 'efnv2s'
#bn_eps = 1e-5
#rst_path = '/kaggle/input/cassava-deeptrane-rst'
#rst_name = ''
optimizer = "Adam"  # Adam AdamW SGD
dropout_ps = [0, 0, 0, 0]
lin_ftrs = [9, 768, 768, 768]
act_head = 'silu'
freeze_for_loading = ['all_but_bn']
freeze = ['none']


from pathlib import Path

_accepted_types = (int, float, str, bool, tuple, list, dict)
cfg = {k: v for k, v in globals().items() if not k.startswith('_') and isinstance(v, _accepted_types)}

cfg["name"] = Path(__file__).stem
cfg["tags"] = cfg["name"].split("_")

# project-dependent settings (deprecated, put in project module)
cfg["curve"] = 'gamma' if 'gamma' in cfg["tags"] else 'beta' if 'beta' in cfg["tags"] else 'free'
cfg["pretrained"] = 'noisy-student' if 'v1' in cfg["arch_name"] else 'imagenet21k-ft1k'
if cfg["epochs"] == 1: cfg["batch_verbose"] = 1
