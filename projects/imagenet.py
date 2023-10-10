from pathlib import Path
import tensorflow as tf


def init(cfg):

    #cfg.meta_csv = cfg.competition_path / 'train.csv'  # label
    cfg.gcs_filter = 'train_tfrecords/*.tfrec'
    if 'tf' in cfg.tags:
        cfg.n_classes = 1000  # pytorch: set by metadata
    #cfg.gcs_paths = {
    #    'cassava-leaf-disease-classification': 'gs://kds-2f1ef51620b1d194f7b5370df991a8cf346870f5e8d86f95e5b81ba3',
    #    }  # set in config

    # Customize data pipeline (see tf_data for definition and defaults)
    cfg.tfrec_format = {'image/encoded': tf.io.FixedLenFeature([], tf.string),
                        #'image/height': tf.io.FixedLenFeature([], tf.int64),
                        #'image/width': tf.io.FixedLenFeature([], tf.int64)},
                        'image/class/label': tf.io.FixedLenFeature([], tf.int64)}
    cfg.data_format = {'image': 'image/encoded',
                       #'height': 'image/height',
                       #'width': 'image/width'},
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


def get_pretrained_model(cfg, strategy):
    if cfg.keep_pretrained_head:
        return get_model_with_pretrained_head(cfg, strategy)
    else:
        from models_tf import get_pretrained_model as default_get_pretrained_model
        return default_get_pretrained_model(cfg, strategy)


def get_model_with_pretrained_head(cfg, strategy):
    import tensorflow as tf
    from tensorflow.keras.layers import Input
    from normalization import Normalization
    #from models_tf import get_bottleneck_params
    assert 'efnv2' in cfg.arch_name, 'Only EfficientnetV2 supported'
    import keras_efficientnet_v2 as efn

    with strategy.scope():

        # Inputs & Pooling
        input_shape = (*cfg.size, 3)
        inputs = [Input(shape=input_shape, name='image')]

        # Body and Feature Extractor
        model_cls = getattr(efn, f'EfficientNetV2{cfg.arch_name[5:].upper()}')
        pretrained_model = model_cls(input_shape=input_shape, 
                                     #num_classes=None,
                                     pretrained=cfg.pretrained)

        embed = pretrained_model(inputs[0])

        # Output
        assert cfg.curve and (cfg.curve == 'free')
        #output = Dense(cfg.channel_size, name='regressor')(embed)
        output = embed
        outputs = [output]

        # Build model
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr, beta_1=cfg.betas[0], beta_2=cfg.betas[1])

        cfg.metrics = cfg.metrics or []

        metrics_classes = {}
        if 'acc' in cfg.metrics:
            metrics_classes['acc'] = tf.keras.metrics.SparseCategoricalAccuracy(name='acc')
        if 'top5' in cfg.metrics:
            metrics_classes['top5'] = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5')
        if 'f1' in cfg.metrics:
            #metrics_classes['f1'] = tfa.metrics.F1Score(num_classes=cfg.n_classes, average='micro', name='F1')
            metrics_classes['f1'] = tf.keras.metrics.F1Score(average='micro', name='F1')
        if 'f2' in cfg.metrics:
            #metrics_classes['f2'] = tfa.metrics.FBetaScore(num_classes=cfg.n_classes, beta=2.0, average='micro', name='F2')
            metrics_classes['f2'] = tf.keras.metrics.FBetaScore(beta=2, average='micro', name='F2')
        if 'macro_f1' in cfg.metrics:
            #metrics_classes['macro_f1'] = tfa.metrics.F1Score(num_classes=cfg.n_classes, average='macro', name='macro_F1')
            metrics_classes['macro_f1'] = tf.keras.metrics.F1Score(average='macro', name='macro_F1')
        if 'curve_rmse' in cfg.metrics:
            from projects.autolevels import TFCurveRMSE
            metrics_classes['curve_rmse'] = TFCurveRMSE(curve=cfg.curve)

        metrics = [metrics_classes[m] for m in cfg.metrics]

        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy' if cfg.classes else 'mean_squared_error',
                      metrics=metrics)

    return model
