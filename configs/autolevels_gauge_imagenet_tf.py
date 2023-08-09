# Setup
project = 'autolevels'
datasets = ['imagenet-1k-tfrecords-ilsvrc2012-part-0', 
            'imagenet-1k-tfrecords-ilsvrc2012-part-1']
gcs_paths = ['gs://kds-dd89dd7e781de3db98a7b5fc06eb08f9ee7f4bb5aa9d4aded39d4caa', 'gs://kds-ae9794a42dd745dac5d4778eed6b1778ba40659b646849382e3c3679']
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
preprocess = 'gamma'
freeze_body = True
freeze_head = True
freeze_preprocess = False
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
