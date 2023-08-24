# Setup
project = 'autolevels'
datasets = ['places365-tfrec-0', 'places365-tfrec-1']
enable_private_datasets = True
gcs_paths = ['gs://kds-b9522d37646f5292f6376eea3f3bd63a8fbc2d8e2bad53f3d7e65b97', 
             'gs://kds-db5957173dabc751d4970483122e29658abc123e9ac350c3ed76dc5b']

#gcs_filters = ['*/*.tfrec', '*/*.tfrec']  # individual filter for each gcs_path
gcs_filter = '*/*.tfrec'
#tfrec_filename_pattern = r'*.tfrec'  # for count_data_items
### TODO: can cases in autolevels.count_data_items be replaced by a tfrec_filename_pattern?

split_by_name = {'valid': 'valid'}  # specify filename substring for 'train' or 'valid'
BGR = False
#preprocess = {'bp': [],
#              'gamma': [],
#              'bp2': []}
#preprocess = {'bp': [0.0, 0.0, 0.0],
#              'a': [0.463185,  0.483004,  0.409333],
#              'b': [0.776565,  0.797041,  0.729030],
#              'bp2': [0.014465,  0.021648, -0.015674]}
freeze_for_loading = ['none']
freeze = ['head']
out_dir = '/kaggle/working'
use_custom_training_loop = False

# Training
seed = 42
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
#save_best = 'loss'
predict_inverse = True
blackpoint_range = (-30, 30)   # x-offset
blackpoint2_range = (-30, 30)  # y-offset
log_gamma_range = [-1.8, 1.8]
mirror_gamma = False
curve3_a_range = (0.2, 1.06) #(0.2, 1.06)(0.3, 0.693) #(0.34, 1.06)  # log_alpha
curve3_beta_range = (0.5, 1)
mirror_beta = False
curve4_loga_range = (-1.8, 0.0)
curve4_b_range = (0.4, 1.2)
mirror_curve4 = True
p_gamma = 0.4                  # probability for using Gamma curve
p_beta = 0.33                  # probability for using Beta PDF rather than Curve4
add_uniform_noise = False       # add uniform noise to mask uint8 discretization [bool|float]
add_jpeg_artifacts = True
sharpness_augment = True
noise_level = 0.01             # random normal noise (augmentation)
augmentation = 'autolevels_aug_tf'

# Model
arch_name = 'efnv2s'
bn_eps = 1e-5
rst_path = '/kaggle/input/autolevels-modelbox'
#rst_name = 'free_efnv2s_r432.h5'
optimizer = "Adam"  # Adam AdamW SGD
wd = 5e-2
dropout_ps = [0, 0, 0, 0]
lin_ftrs = [9, 768, 768, 768]
act_head = 'silu'
freeze_for_loading = ['head']
freeze = ['head']


from pathlib import Path

_accepted_types = (int, float, str, bool, tuple, list, dict)
cfg = {k: v for k, v in globals().items() if not k.startswith('_') and isinstance(v, _accepted_types)}

cfg["name"] = Path(__file__).stem
cfg["tags"] = cfg["name"].split("_")

# project-dependent settings (deprecated, put in project module)
cfg["curve"] = 'gamma' if 'gamma' in cfg["tags"] else 'beta' if 'beta' in cfg["tags"] else 'free'
cfg["pretrained"] = 'noisy-student' if 'v1' in cfg["arch_name"] else 'imagenet21k-ft1k'
if cfg["epochs"] == 1: cfg["batch_verbose"] = 1
