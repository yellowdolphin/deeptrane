# Setup
project = 'autolevels'
out_dir = '/kaggle/working'  # job subdir is automatically appended
filetype = 'JPEG'
meta_csv = '/content/meta.csv'  # for colab only

# Training
num_folds = 20
use_folds = [0]
train_on_all = False
frac = [0.3, 1.0]
size = (384, 384)
presize = 2.0                  # only used if use_batch_tfms
antialias = False
interpolation = 'NEAREST'
predict_inverse = True
blackpoint_range = (-100, 10)  # x-offset
blackpoint2_range = (-75, 10)  # y-offset
whitepoint_sigma = 80
clip_target_blackpoint = True
log_gamma_range = [-1.8, 1.8]
curve3_conditional_a_range = (-1.0, 0.30)
curve3_beta_range = (0.5, 1.35)
mirror_beta = False
curve4_loga_range = (-2.0, 0.5)
curve4_conditional_logb_range = (0.00, 0.47)  # logb = logb_0 + logb_range * (loga - loga_0)
curve4_conditional_logb_offsets = (-0.8, -3.93)  # (logb_0, loga_0)
mirror_curve4 = False
p_gamma = 0.3                  # probability for using Gamma curve
p_beta = 0.5                   # probability for using Beta PDF rather than Curve4
add_uniform_noise = False      # add uniform noise to mask uint8 discretization [bool|float]
add_jpeg_artifacts = True
sharpness_augment = True
noise_level = 0.01             # random normal noise (augmentation)
resize_before_jpeg = True
augmentation = 'tfms_004'      # ignored if use_batch_tfms
use_batch_tfms = False
n_replicas = 1
use_dp = False                 # slower on 2 T4 than on 1
improve_color_loss = 0         # weight of auxiliary loss to improve colors
metrics = ['curve_rmse']
no_macro_metrics = True        # otherwise slow valid + 8-TPU-issue
bs = 32
n_acc = 8
batch_verbose = 1
lr_head = 1e-5
one_cycle = False
div_factor = 5                            # default: 25, from Chest14: 1
pct_start = 0.25                           # default: 0.3, from Chest14: 0.6, pipeline1: 0
save_best = None  # 'train_loss'

# Model
arch_name = 'tiny_vit_21m_384.dist_in22k_ft_in1k'
scale_output_layer = 1.0
use_gem = False
bn_eps = 1e-5
rst_path = '/kaggle/input/rst-autolevels-train3/job_0'
rst_name = 'autolevels_imagenet_free_job0_fold0_ep13'
epochs = 15
reset_opt = False
optimizer = "Adam"  # Adam AdamW SGD
dropout_ps = [0, 0, 0, 0]
lin_ftrs = [30, 768, 768, 768]
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
