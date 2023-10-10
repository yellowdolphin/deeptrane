# Setup
project = 'imagenet'
datasets = ['imagenet-1k-tfrecords-ilsvrc2012-part-0', 
            'imagenet-1k-tfrecords-ilsvrc2012-part-1']  # target_map defined in configs/imagenet_aug_tf.py
gcs_paths = ['gs://kds-b82d4220a658d96cc9944a562792277dba4f6d8a4367faf42c01ed25', 
             'gs://kds-8b208708b137ae5643e8cd144348c8b467a6cd7dad15ab57e3e3de0e']
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
metrics = ['acc']
no_macro_metrics = True  # otherwise slow valid + 8-TPU-issue
bs = 64
steps_per_execution = 1  # increase for performance (check callbacks, training behavior)
epochs = 5
batch_verbose = 1
lr = 1e-3
one_cycle = True
div_factor = 5                            # default: 25, from Chest14: 1
pct_start = 0.25                           # default: 0.3, from Chest14: 0.6, pipeline1: 0
lr_min = 1e-6
save_best = 'loss'
augmentation = 'imagenet_aug_tf'

# Model
arch_name = 'efnv2s'
#bn_eps = 1e-5
#rst_path = '/kaggle/input/cassava-deeptrane-rst'
#rst_name = ''
optimizer = "Adam"  # Adam AdamW SGD
#dropout_ps = [0, 0, 0, 0]
#lin_ftrs = [24, 768, 768, 768]
#act_head = 'silu'
#freeze_for_loading = ['none']
#freeze = ['none']
keep_pretrained_head = True


from pathlib import Path

_accepted_types = (int, float, str, bool, tuple, list, dict)
cfg = {k: v for k, v in globals().items() if not k.startswith('_') and isinstance(v, _accepted_types)}

cfg["name"] = Path(__file__).stem
cfg["tags"] = cfg["name"].split("_")

# project-dependent settings (deprecated, put in project module)
cfg["curve"] = 'gamma' if 'gamma' in cfg["tags"] else 'beta' if 'beta' in cfg["tags"] else 'free'
cfg["pretrained"] = 'noisy-student' if 'v1' in cfg["arch_name"] else 'imagenet21k-ft1k'
if cfg["epochs"] == 1: cfg["batch_verbose"] = 1
