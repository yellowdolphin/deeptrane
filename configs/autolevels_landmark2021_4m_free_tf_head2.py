# Setup
project = 'autolevels'
datasets = [f'google-landmark-4m-tfrecords-part-{i}' for i in (0, 1, 2, 4, 7, 8, 9)]
gcs_paths = [
    'gs://kds-aa92dd3ef86be8cd790806d7b1e46521a8bf7bee866f86fa4145f5ba', 
    'gs://kds-c36d6ae7f6a6495603f5604ef8cb7b34091a940d6d7a00f32e32d4ea', 
    'gs://kds-31c206fa1ec9309046fe9144d4294f93222f05a7b83814a424789457', 
    'gs://kds-4e48b6774cb45d15755bc191dff627e668159b843b27b6cbc5c396f9', 
    'gs://kds-9237ac2f61cecec928dcbc6debc721c8db72a7a95f51cb6ec7e487d2', 
    'gs://kds-96ea77240b2e9bd5e54db6017f8af46319398a86ba502e9c54e4b636', 
    'gs://kds-a46098b7f7469288f109cfb54507227ec96137d41c9aa25ce42bafbd']
gcs_filter = '*.tfrec'
tfrec_filename_pattern = None
BGR = False
#preprocess = {'bp': [],
#              'gamma': [],
#              'bp2': []}
preprocess = {'bp': [0.0, 0.0, 0.0],
              'a': [0.463185,  0.483004,  0.409333],
              'b': [0.776565,  0.797041,  0.729030],
              'bp2': [0.014465,  0.021648, -0.015674]}
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
presize = 1.0            # scale images to size * presize if tfrec with varying sizes contains no height/width
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
rst_name = 'free_efnv2s_r608.h5'
rst_epoch = 48
optimizer = "Adam"  # Adam AdamW SGD
dropout_ps = [0, 0, 0, 0]
lin_ftrs = [12, 768, 768, 768]
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
