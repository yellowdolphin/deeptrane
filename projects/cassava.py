from pathlib import Path


def init(cfg):
    if cfg.filetype == 'jpg':
        assert cfg.size[0] == cfg.size[1]
        size = 128 if (cfg.size[0] <= 128) else 256 if (cfg.size[0] <= 256) else 512
        cfg.image_root = Path(f'/kaggle/input/cassava-jpeg-{size}x{size}/kaggle/train_images_jpeg')
        if cfg.cloud == 'drive':
            #cfg.image_root = Path(f'/content/cassava-jpeg-{size}x{size}/kaggle/train_images_jpeg')
            cfg.image_root = Path(f'/tmp/cassava-jpeg-{size}x{size}/kaggle/train_images_jpeg')

    cfg.meta_csv = Path('/kaggle/input/cassava-leaf-disease-classification/train.csv')  # label
    if cfg.cloud == 'drive':
        cfg.meta_csv = Path(f'/content/gdrive/MyDrive/{cfg.project}/train.csv')
    cfg.gcs_filter = 'train_tfrecords/*.tfrec'
    if 'tf' in cfg.tags:
        cfg.n_classes = 5  # pytorch: set by metadata
    #cfg.gcs_paths = {
    #    'cassava-leaf-disease-classification': 'gs://kds-2f1ef51620b1d194f7b5370df991a8cf346870f5e8d86f95e5b81ba3',
    #    }  # set in config

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
    #print("DEBUG: truncating train+valid data to full batches!")
    #n_replicas = cfg.n_replicas or 1
    ##new_len = len(df) - len(df) % (cfg.num_folds * cfg.bs * n_replicas)
    #new_len = 1 * cfg.num_folds * cfg.bs * n_replicas  # minimal training
    #df = df.iloc[:new_len].copy()

    return df
