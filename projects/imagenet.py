from pathlib import Path
import tensorflow as tf
from models_tf import set_trainable, set_bn_parameters


def init(cfg):

    #cfg.meta_csv = cfg.competition_path / 'train.csv'  # label
    cfg.gcs_filter = 'train_tfrecords/*.tfrec'
    if 'tf' in cfg.tags:
        cfg.classes = True    # its bool must validate True to get the right loss function
        cfg.n_classes = 1000  # pytorch: set by metadata

    # Customize data pipeline (see tf_data for definition and defaults)
    cfg.tfrec_format = {'image/encoded': tf.io.FixedLenFeature([], tf.string),
                        'image/class/label': tf.io.FixedLenFeature([], tf.int64)}
    cfg.data_format = {'image': 'image/encoded',
                       'target': 'image/class/label'}
    cfg.inputs = ['image']
    cfg.targets = ['target']


def count_examples_in_tfrec(fn):
    count = 0
    for _ in tf.data.TFRecordDataset(fn):
        count += 1
    return count


def count_data_items(filenames, tfrec_filename_pattern=None):
    if '-of-' in filenames[0]:
        # Imagenet: Got number of items from idx files
        return sum(391 if 'validation' in fn else 1252 for fn in filenames)
    if 'FlickrFaces' in filenames[0]:
        # flickr: number of items in filename
        return sum(int(fn[-10:-6]) for fn in filenames)
    if 'coco' in filenames[0]:
        return sum(159 if 'train/coco184' in fn else 499 if 'val/coco7' in fn else 
                   642 if '/train/' in fn else 643 
                   for fn in filenames)
    if 'landmark-2021-gld2' in filenames[0]:
        return sum(
            203368 if 'gld2-0' in fn else 203303 if 'gld2-1' in fn else 
            203376 if 'gld2-2' in fn else 203316 if 'gld2-4' in fn else 
            203415 if 'gld2-7' in fn else 203400 if 'gld2-8' in fn else
            203321 if 'gld2-9' in fn else 0  # gld2-3 is corrupt, 5 and 6 missing
            for fn in filenames)
    if all('train_' in fn for fn in filenames) and len(filenames) in [45, 6]:
        # landmark2021: no subfolders or identifiable string in urls
        # if too large, valid raises out-of-range error
        return 1448943 if len(filenames) == 45 else 193665
    if all('train_' in fn for fn in filenames) and len(filenames) in [46, 4]:
        # landmark2020: no subfolders or identifiable string in urls
        # if too large, valid raises out-of-range error
        # all 31610 except train_20 ...31 (31609)
        return 31610 * 46 if len(filenames) == 46 else 31609 * 4
    if Path(filenames[0]).stem[3] == '-':
        # places365-tfrec: number of items in filename
        return sum(int(Path(fn).stem[4:]) for fn in filenames)
    if tfrec_filename_pattern is not None:
        "Infer number of items from tfrecord file names."
        tfrec_filename_pattern = tfrec_filename_pattern or r"-([0-9]*)\."
        pattern = re.compile(tfrec_filename_pattern)
        n = [int(pattern.search(fn).group(1)) for fn in filenames if pattern.search(fn)]
        if len(n) < len(filenames):
            print(f"WARNING: only {len(n)} / {len(filenames)} urls follow the convention:")
            for fn in filenames:
                print(fn)
        return sum(n)
    else:
        if True:
            # count them (slow)
            print(f'Counting examples in {len(filenames)} urls:')
            total_count = 0
            for i, fn in enumerate(filenames):
                n = count_examples_in_tfrec(fn)
                #print(f"   {i:3d}", Path(fn).parent, n)
                print(f"   {i:3d}", fn, n)
                total_count += n
            return total_count
        else:
            raise NotImplementedError(f'autolevels.count_data_items: filename not recognized: {filenames[0]}')
