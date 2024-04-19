# Setup
project = 'autolevels'
out_dir = '/kaggle/working'
filetype = 'jpg'
meta_csv = '/kaggle/input/autolevels-modelbox/coco2017.csv'  # for colab only

# Training
num_folds = 5
use_folds = [0]
train_on_all = False
size = (384, 384)
presize = 2.0                  # only used if use_batch_tfms
antialias = False
interpolation = 'NEAREST'
predict_inverse = True
blackpoint_range = (-30, 30)   # x-offset
blackpoint2_range = (-30, 30)  # y-offset
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
resize_before_jpeg = True
augmentation = 'tfms_004'      # ignored if use_batch_tfms
use_batch_tfms = False
n_replicas = 8
use_dp = False                 # slower on 2 T4 than on 1
metrics = ['curve_rmse']
no_macro_metrics = True        # otherwise slow valid + 8-TPU-issue
bs = 32
epochs = 1
batch_verbose = 1
lr_head = 1e-2
one_cycle = True
div_factor = 5                            # default: 25, from Chest14: 1
pct_start = 0.25                           # default: 0.3, from Chest14: 0.6, pipeline1: 0
save_best = 'train_loss'

# Model
arch_name = 'rexnet_130'
scale_output_layer = 1.0
use_gem = False
bn_eps = 1e-5
rst_path = '/kaggle/input/autolevels-modelbox'
#rst_name = ''
optimizer = "Adam"  # Adam AdamW SGD
dropout_ps = [0, 0, 0, 0]
lin_ftrs = [12, 768, 768, 768]
act_head = 'SiLU'


from pathlib import Path

_accepted_types = (int, float, str, bool, tuple, list)
cfg = {k: v for k, v in globals().items() if not k.startswith('_') and isinstance(v, _accepted_types)}

cfg["name"] = Path(__file__).stem
cfg["tags"] = cfg["name"].split("_")

# project-dependent settings (deprecated, put in project module)
cfg["curve"] = 'gamma' if 'gamma' in cfg["tags"] else 'beta' if 'beta' in cfg["tags"] else 'free'
cfg["lr_bn"] = cfg["lr_head"]
cfg["lr_body"] = cfg["lr_head"]
if cfg["epochs"] == 1: cfg["batch_verbose"] = 10
