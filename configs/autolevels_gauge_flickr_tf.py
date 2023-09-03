# Setup
project = 'autolevels'
datasets = ['flickrfacestfrecords']
gcs_paths = ['gs://kds-c922a5927286da71f584b1434bd5b31d19e55d463b3f38f39190dc8d']
gcs_filters = ['FlickrFaces*.tfrec']
tfrec_filename_pattern = None
BGR = False
#preprocess = {'bp': [],
#              'gamma': [],
#              'bp2': []}
#preprocess = {'bp': [0.0, 0.0, 0.0],
#              'a': [0.463185,  0.483004,  0.409333],
#              'b': [0.776565,  0.797041,  0.729030],
#              'bp2': [0.014465,  0.021648, -0.015674]}
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
predict_inverse = False
blackpoint_range = (0, 0)   # x-offset
blackpoint2_range = (0, 0)  # y-offset
log_gamma_range = [0, 0]
mirror_gamma = False
p_gamma = 1.0                  # probability for using Gamma curve
p_beta = 0.0                   # probability for using Beta PDF rather than Curve4
curve3_a_range = (1, 1) #(0.2, 1.06)(0.3, 0.693) #(0.34, 1.06)  # log_alpha
curve3_beta_range = (1, 1)
curve4_loga_range = (0, 0)
curve4_b_range = (1, 1)
add_uniform_noise = False      # add uniform noise to mask uint8 discretization [bool|float]
add_jpeg_artifacts = True
sharpness_augment = True
noise_level = 0.01             # random normal noise (augmentation)
augmentation = 'autolevels_aug_tf'

# Model
arch_name = 'efnv2s'
bn_eps = 1e-5
rst_path = '/kaggle/input/autolevels-modelbox'
#rst_name = ''
optimizer = "Adam"  # Adam AdamW SGD
dropout_ps = [0, 0, 0, 0]
lin_ftrs = [24, 768, 768, 768]
act_head = 'silu'
freeze_for_loading = ['none']
freeze = ['body', 'head']
preprocess = 'gamma'


from pathlib import Path

_accepted_types = (int, float, str, bool, tuple, list, dict)
cfg = {k: v for k, v in globals().items() if not k.startswith('_') and isinstance(v, _accepted_types)}

cfg["name"] = Path(__file__).stem
cfg["tags"] = cfg["name"].split("_")

# project-dependent settings (deprecated, put in project module)
cfg["curve"] = 'gamma' if 'gamma' in cfg["tags"] else 'beta' if 'beta' in cfg["tags"] else 'free'
cfg["pretrained"] = 'noisy-student' if 'v1' in cfg["arch_name"] else 'imagenet21k-ft1k'
if cfg["epochs"] == 1: cfg["batch_verbose"] = 1
