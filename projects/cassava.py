from pathlib import Path

gcs_paths = {
    'cassava-leaf-disease-classification': 'gs://kds-2f1ef51620b1d194f7b5370df991a8cf346870f5e8d86f95e5b81ba3',
    }

def init(cfg):
    cfg.competition_path = Path('/kaggle/input/cassava-leaf-disease-classification')
    if cfg.cloud == 'drive':
        cfg.competition_path = Path(f'/content/gdrive/MyDrive/{cfg.project}')

    if cfg.filetype == 'jpg':
        assert cfg.size[0] == cfg.size[1]
        size = cfg.size[0]
        cfg.image_root = Path(f'/kaggle/input/cassava-jpeg-{size}x{size}/kaggle/train_images_jpeg')
        if cfg.cloud == 'drive':
            #cfg.image_root = Path(f'/content/cassava-jpeg-{size}x{size}/kaggle/train_images_jpeg')
            cfg.image_root = Path(f'/tmp/cassava-jpeg-{size}x{size}/kaggle/train_images_jpeg')

    #elif cfg.filetype == 'tfrec':
    #    cfg.image_root = cfg.competition_path / 'train_tfrecords'

    cfg.meta_csv = cfg.competition_path / 'train.csv'  # label
    cfg.gcs_filter = 'train_tfrecords/*.tfrec'
    cfg.n_classes = 5
    cfg.gcs_paths = gcs_paths

    # Customize data pipeline (see tf_data for definition and defaults)
    #cfg.tfrec_format = {
    #    'image': tf.io.FixedLenFeature([], tf.string),
    #    'box': tf.io.FixedLenFeature([4], tf.int64),
    #    'target': tf.io.FixedLenFeature([], tf.int64)}
    #cfg.data_format = {
    #    'image': 'image',
    #    #'bbox': 'box',
    #    'target': 'target'}
    #cfg.inputs = ['image']
    #cfg.targets = ['target']


def add_category_id(df, cfg):
    return df.rename(columns={'label': 'category_id'})


def add_image_id(df, cfg):
    df['image_id'] = df.image_id.str.split('.').str[0]

    # DEBUG: truncate data to multiple of TPU global_batch_size * num_folds
    n_replicas = cfg.n_replicas or 1
    new_len = len(df) - len(df) % (cfg.num_folds * cfg.bs * n_replicas)
    df = df.iloc[:new_len].copy()

    return df
