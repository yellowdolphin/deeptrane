# Setup
project = 'autolevels'
datasets = [f'google-landmark-tfrecords-512-{i}' for i in range(1, 7)]
gcs_paths = ['gs://kds-312b8e8d30ec0d08b626be22aa6e33dff085ebcfa0923ba743379431', 'gs://kds-875b41c4d9a800bff52ebb306948b58c12990b98c3d475931693559e', 'gs://kds-d05de08723713ac86b96d902c091c90ea371ed9d8ab6311949965b46', 'gs://kds-ba90cd7fd19316762ed1bd392e281ca5b13e12d9bc6bb376578748a8', 'gs://kds-79c99aafaa6c48e32b8cdea47434559ef64ec356530c1b9cc6f15d60', 'gs://kds-9ea89c976c4ee7d0116bc7b82e063004d2af987da628896c48183840']
gcs_filter = '*.tfrec'
tfrec_filename_pattern = None
BGR = True
preprocess = {
    'bp': [0.040800,  0.060259,  0.024672],
    'gamma': [1.026049,  1.054437,  1.010604],
    'bp2': [-0.012288, -0.018215, -0.008333],
    'bp_ref': [0.0085, 0.0232, 0.0068],
    'gamma_ref': [0.9992, 1.0180, 0.9982],
    'bp2_ref': [0.0095, 0.0031, 0.0111],
}
#preprocess = {'bp': [-0.111739,  0.033335, -0.000446],
#              'a': [0.446492,  0.536055,  0.487654],
#              'b': [0.773841,  0.842389,  0.802490],
#              'bp2': [0.061021, -0.012432, -0.006443]}
freeze_for_loading = ['none']
freeze = ['head']
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
add_uniform_noise = False      # add uniform noise to mask uint8 discretization [bool|float]
add_jpeg_artifacts = True
sharpness_augment = True
noise_level = 0.01             # random normal noise (augmentation)
resize_before_jpeg = True
augmentation = 'autolevels_aug_tf'
catmix = False                 # compose input of 4 images

# Model
arch_name = 'efnv2s'
#bn_eps = 1e-5
rst_path = '/kaggle/input/autolevels-modelbox'
#rst_name = 'free_efnv2s_r608.h5'
#rst_epoch = 48
optimizer = "Adam"  # Adam AdamW SGD
dropout_ps = [0, 0, 0, 0]
lin_ftrs = [24, 768, 768, 768]
act_head = 'silu'


from pathlib import Path

_accepted_types = (int, float, str, bool, tuple, list, dict)
cfg = {k: v for k, v in globals().items() if not k.startswith('_') and isinstance(v, _accepted_types)}

cfg["name"] = Path(__file__).stem
cfg["tags"] = cfg["name"].split("_")

# project-dependent settings (deprecated, put in project module)
cfg["curve"] = 'gamma' if 'gamma' in cfg["tags"] else 'beta' if 'beta' in cfg["tags"] else 'free'
cfg["pretrained"] = 'noisy-student' if 'v1' in cfg["arch_name"] else 'imagenet21k-ft1k'
if cfg["epochs"] == 1: cfg["batch_verbose"] = 1
