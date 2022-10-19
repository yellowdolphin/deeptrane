"""
Models from tf.keras.applications suck. 
Efficientnet V1 works best from `efficientnet` library.
Efficientnet V2 works best from keras-efficientnet-v2.
Other models can be obtained from tfimm or tfhub libraries, 
but there are issues:

### tfimm models
- convnext_base_384_in22ft1k OK, 42 h/ep CPU, 75 sec/ep TPU
- cait_xs24_384 requires size=384, 32 h/ep CPU
- cait_s24_224 requires size=224, 13 h/ep CPU
- vit_base_patch16_384 requires size=384, 43 h/ep CPU
- vit_base_patch8_224 requires size=224, bs=8, 61 h/ep CPU
- swin_base_patch4_window12_384 requires size=384, OK in CPU/GPU, error on TPU
- swin_base_patch4_window7_224_in22k OK in CPU/GPU, error on TPU (XLA cannot infer shape for model/swin_transformer_1/layers/0/blocks/0/attn/qkv/Tensordot operation)
- deit_base_distilled_patch16_384
    - First dimension of predictions 4 must match length of targets 2

### tfhub models
- efnv2 OK, 7 h/ep CPU (224)
- bit_m-r50x1 OK, 7 h/ep CPU (224), 15 h/ep CPU (384)
- bit_1k-*
    - save_weights error
- vit_b8 requires size=224, fast but Errors on TPU in/after validation
    - save_weights error
- convnext_base_21k_1k_224_fe
    - UnimplementedError:  Fused conv implementation does not support grouped convolutions for now.
"""
import math

import tensorflow as tf
from utils.general import quietly_run

# if cfg.arch_name.startswith('efnv1'):
#     import efficientnet.tfkeras as efn
#     EFN = {'efnv1b0': efn.EfficientNetB0, 'efnv1b1': efn.EfficientNetB1, 
#            'efnv1b2': efn.EfficientNetB2, 'efnv1b3': efn.EfficientNetB3,
#            'efnv1b4': efn.EfficientNetB4, 'efnv1b5': efn.EfficientNetB5, 
#            'efnv1b6': efn.EfficientNetB6, 'efnv1b7': efn.EfficientNetB7}

# if cfg.arch_name.startswith('efnv2'):
#     import keras_efficientnet_v2 as efn
#     EFN = {'efnv2s': efn.EfficientNetV2S, 'efnv2m': efn.EfficientNetV2M,
#            'efnv2l': efn.EfficientNetV2L, 'efnv2xl': efn.EfficientNetV2XL}

TFHUB = {
    'hub_efnv2s': "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_s/feature_vector/2",
    'hub_efnv2m': "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_m/feature_vector/2",
    'hub_efnv2l': "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_l/feature_vector/2",
    'hub_efnv2xl': "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet21k_ft1k_xl/feature_vector/2",
    'bit_m-r50x1': "https://tfhub.dev/google/bit/m-r50x1/1",
    'bit_m-r50x3': "https://tfhub.dev/google/bit/m-r50x3/1",
    'bit_m-r101x1': "https://tfhub.dev/google/bit/m-r101x1/1",
    'bit_m-r101x3': "https://tfhub.dev/google/bit/m-r101x3/1",
    'bit_m-r152x4': "https://tfhub.dev/google/bit/m-r152x4/1",
    #'bit_1k-r50x1_224': "https://tfhub.dev/sayakpaul/distill_bit_r50x1_224_feature_extraction/1",
    #'bit_1k-r152x2_384': "https://tfhub.dev/sayakpaul/bit_r152x2_384_feature_extraction/1",
    #'vit_b8': "https://tfhub.dev/sayakpaul/vit_b8_classification/1",
    #'convnext_base_21k_1k_384_fe': "https://tfhub.dev/sayakpaul/convnext_base_21k_1k_384_fe/1",
    #'convnext_base_21k_1k_224_fe': "https://tfhub.dev/sayakpaul/convnext_base_21k_1k_224_fe/1",
}


class GeMPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, p=1., train_p=False):
        super().__init__()
        if train_p:
            self.p = tf.Variable(p, dtype=tf.float32)
        else:
            self.p = p
        self.eps = 1e-6

    def call(self, inputs: tf.Tensor, **kwargs):
        inputs = tf.clip_by_value(inputs, clip_value_min=1e-6, clip_value_max=tf.reduce_max(inputs))
        inputs = tf.pow(inputs, self.p)
        inputs = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        inputs = tf.pow(inputs, 1./self.p)
        return inputs


class CustomTrainStep(tf.keras.Model):
    def __init__(self, n_acc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_acc = tf.constant(n_acc, dtype=tf.int32)
        self.n_grad_steps = tf.Variable(0, dtype=tf.int32, trainable=False)
        #for w in self.weights:            # safer, but needs more memory
        for w in self.trainable_weights:  # unfreezing layers may break grad accumulation
            w.grad = tf.Variable(tf.zeros_like(w), trainable=False)
        
    @property
    def trainable_grads(self):
        return [w.grad for w in self.trainable_weights]
    
    def zero_grads(self):
        for g in self.trainable_grads:
            g.assign(tf.zeros_like(g))

    def add_grads(self, new_grads):
        for x, y in zip(self.trainable_grads, new_grads):
            x.assign_add(y)

    @tf.function
    def optimizer_step(self):
        self.optimizer.apply_gradients(zip(self.trainable_grads, self.trainable_variables))
        self.zero_grads()
        return 0
    
    def train_step(self, data):
        # Supports multiple losses (outputs/labels) y = [y0, y1, ...]
        #x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)  # tf>2.4
        x, y = data
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            #loss = self.compiled_loss(y, y_pred, sample_weight, regularization_losses=self.losses)  # tf>2.4
            #loss = self.compute_loss(x, y, y_pred, sample_weight)  # tf>2.7

        self.add_grads(tape.gradient(loss, self.trainable_variables))
        self.n_grad_steps.assign(
            tf.cond(tf.equal(self.n_grad_steps, self.n_acc), 
                    self.optimizer_step, lambda: self.n_grad_steps + 1))
        #return self.compute_metrics(x, y, y_pred, sample_weight)  # tf>2.7
        self.compiled_metrics.update_state(y, y_pred)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics


class SequentialWithGrad(tf.keras.Sequential):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #for w in self.weights:            # safer, but needs more memory
        for w in self.trainable_weights:  # unfreezing layers may break grad accumulation
            w.grad = tf.Variable(tf.zeros_like(w), trainable=False)
        
    @property
    def trainable_grads(self):
        return [w.grad for w in self.trainable_weights]
    
    def zero_grads(self):
        for g in self.trainable_grads:
            g.assign(tf.zeros_like(g))

    def add_grads(self, new_grads):
        for x, y in zip(self.trainable_grads, new_grads):
            x.assign_add(y)


class ModelWithGrad(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #for w in self.weights:            # safer, but needs more memory
        for w in self.trainable_weights:  # unfreezing layers may break grad accumulation
            w.grad = tf.Variable(tf.zeros_like(w), trainable=False)
        
    @property
    def trainable_grads(self):
        return [w.grad for w in self.trainable_weights]
    
    def zero_grads(self):
        for g in self.trainable_grads:
            g.assign(tf.zeros_like(g))

    def add_grads(self, new_grads):
        for x, y in zip(self.trainable_grads, new_grads):
            x.assign_add(y)


class ArcMarginProductSubCenter(tf.keras.layers.Layer):
    '''
    Implements large margin arc distance.

    References:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
        https://github.com/haqishen/Google-Landmark-Recognition-2020-3rd-Place-Solution/

    Sub-center version:
        for k > 1, the embedding layer can learn k sub-centers per class
    '''
    def __init__(self, n_classes, s=30, m=0.50, k=1, easy_margin=False,
                 ls_eps=0.0, **kwargs):

        super(ArcMarginProductSubCenter, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.k = k
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(math.pi - m)
        self.mm = tf.math.sin(math.pi - m) * m

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'k': self.k,
            'ls_eps': self.ls_eps,
            'easy_margin': self.easy_margin,
        })
        return config

    def build(self, input_shape):

        super(ArcMarginProductSubCenter, self).build(input_shape[0])

        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes * self.k),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True)

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine_all = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0)
        )
        if self.k > 1:
            cosine_all = tf.reshape(cosine_all, [-1, self.n_classes, self.k])
            cosine = tf.math.reduce_max(cosine_all, axis=2)
        else:
            cosine = cosine_all
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class AddMarginProductSubCenter(tf.keras.layers.Layer):
    """
    Add the subcenter DOF but keep all other properties of AddMarginProduct (my idea)
    https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/blob/master/src/modeling/metric_learning.py
    Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        k: number of subcenters
        cos(theta) - m
    """

    def __init__(self, n_classes, s=30.0, m=0.40, k=3, **kwargs):
        _ = kwargs.pop('easy_margin', None)
        super(AddMarginProductSubCenter, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.k = k

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'k': self.k})
        return config

    def build(self, input_shape):
        super(AddMarginProductSubCenter, self).build(input_shape[0])
        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes * self.k),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True)
    
    def call(self, inputs):
        input, label = inputs
        label = tf.cast(label, dtype=tf.int32)
        cosine_all = tf.matmul(tf.math.l2_normalize(input, axis=1), 
                               tf.math.l2_normalize(self.W, axis=0))
        if self.k == 1:
            cosine = cosine_all
        else:
            cosine_all = tf.reshape(cosine_all, [-1, self.n_classes, self.k])
            cosine = tf.math.reduce_max(cosine_all, axis=2)
        phi = cosine - self.m
        one_hot = tf.cast(tf.one_hot(label, depth=self.n_classes), dtype=cosine.dtype)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


def get_margin(cfg):
    # Adaptive margins for each target class (range: cfg.margin_min ... cfg.margin_maxbe)
    # should be defined in project.
    m = cfg.adaptive_margin or 0.3

    if cfg.arcface == 'ArcMarginProduct':
        return ArcMarginProductSubCenter(cfg.n_classes, m=m, k=cfg.subcenters or 1, easy_margin=cfg.easy_margin,
            name=f'head/{cfg.arcface}', dtype='float32')

    if cfg.arcface == 'AddMarginProduct':
        return AddMarginProductSubCenter(cfg.n_classes, m=m, k=cfg.subcenters or 1,
            name=f'head/{cfg.arcface}', dtype='float32')

    raise ValueError(f'ArcFace type {cfg.arcface} not supported')


def BatchNorm(bn_type):
    if bn_type == 'batch_norm': return tf.keras.layers.BatchNormalization()
    if bn_type == 'sync_bn': return tf.keras.layers.experimental.SyncBatchNormalization()
    if bn_type == 'layer_norm': return tf.keras.layers.LayerNormalization()  # bad valid, nan loss
    #if bn_type == 'instance_norm':
    #    import tensorflow_addons as tfa
    #    return tfa.layers.InstanceNormalization()  # nan loss
    if bn_type == 'instance_norm': return tf.keras.layers.BatchNormalization(virtual_batch_size=cfg.bs)


def check_model_inputs(cfg, model):
    for inp in model.inputs:
        assert inp.name in cfg.data_format, f'"{inp.name}" (model.inputs) missing in cfg.data_format'


def get_pretrained_model(cfg, strategy, inference=False):

    # Imports
    import tensorflow as tf

    if cfg.arch_name.startswith('efnv1'):
        import efficientnet
        import efficientnet.tfkeras as efn
        model_cls = getattr(efn, f'EfficientNetB{cfg.arch_name[6:]}')
        print("efficientnet:", efficientnet.__version__)
    elif cfg.arch_name.startswith('efnv2'):
        import keras_efficientnet_v2 as efn
        model_cls = getattr(efn, f'EfficientNetV2{cfg.arch_name[5:].upper()}')
        print("keras_efficientnet_v2:", efn.__version__)
    elif cfg.arch_name in TFHUB:
        pass
    else:
        import tfimm
        print("tfimm:", tfimm.__version__)
        if cfg.list_models:
            print(tfimm.list_models(pretrained="timm"))

    with strategy.scope():

        # Inputs
        input_shape = (*cfg.size, 3)
        inputs = [tf.keras.layers.Input(shape=input_shape, name='image')]
        if cfg.arcface and not inference: 
            inputs.append(tf.keras.layers.Input(shape=(), name='target'))

        # Body
        efnv1 = cfg.arch_name.startswith('efnv1')
        efnv2 = cfg.arch_name.startswith('efnv2')
        tfhub = cfg.arch_name in TFHUB

        pretrained_model = (
            model_cls(weights=cfg.pretrained, input_shape=input_shape, include_top=False) if efnv1 else
            model_cls(input_shape=input_shape, num_classes=0, pretrained=cfg.pretrained) if efnv2 else
            hub.KerasLayer(TFHUB[cfg.arch_name], trainable=True) if tfhub else
            tfimm.create_model(cfg.arch_name, pretrained="timm", nb_classes=0))

        if cfg.sync_bn:
            pretrained_model = replace_bn_layers(pretrained_model, 
                                                 tf.keras.layers.experimental.SyncBatchNormalization,
                                                 keep_weights=True)
        elif cfg.instance_norm:
            pretrained_model = replace_bn_layers(pretrained_model,
                                                 tf.keras.layers.BatchNormalization,
                                                 keep_weights=True,
                                                 virtual_batch_size=1)

        # Head(s)
        if efnv1:
            x = pretrained_model(inputs[0])
            if cfg.pool == 'flatten':
                embed = tf.keras.layers.Flatten()(x)
            elif cfg.pool == 'fc':
                embed = tf.keras.layers.Flatten()(x)
                embed = tf.keras.layers.Dropout(0.1)(embed)
                embed = tf.keras.layers.Dense(1024)(embed)
            elif cfg.pool == 'gem':
                embed = GeMPoolingLayer(train_p=True)(x)
            elif cfg.pool == 'concat':
                embed = tf.keras.layers.concatenate([tf.keras.layers.GlobalAveragePooling2D()(x),
                                                     tf.keras.layers.GlobalAveragePooling2D()(x)])
            elif cfg.pool == 'max':
                embed = tf.keras.layers.GlobalMaxPooling2D()(x)
            else:
                embed = tf.keras.layers.GlobalAveragePooling2D()(x)
            
        elif efnv2:
            x = pretrained_model(inputs[0])
            if cfg.pool == 'flatten':
                embed = tf.keras.layers.Flatten()(x)
            elif cfg.pool == 'fc':
                embed = tf.keras.layers.Flatten()(x)
                embed = tf.keras.layers.Dropout(0.1)(embed)
                embed = tf.keras.layers.Dense(1024)(embed)
            elif cfg.pool == 'gem':
                embed = GeMPoolingLayer(train_p=True)(x)
            elif cfg.pool == 'concat':
                embed = tf.keras.layers.concatenate([tf.keras.layers.GlobalAveragePooling2D()(x),
                                                     tf.keras.layers.GlobalAveragePooling2D()(x)])
            elif cfg.pool == 'max':
                embed = tf.keras.layers.GlobalMaxPooling2D()(x)
            else:
                embed = tf.keras.layers.GlobalAveragePooling2D()(x)

        elif tfhub:
            # tfhub models cannot be modified => Pooling cannot be changed!
            assert cfg.pool in [None, False, 'avg', ''], 'tfhub model, no custom pooling supported!'
            print(f"{cfg.arch_name} from tfhub")
            embed = pretrained_model(inputs[0])

        else:
            print(f"{cfg.arch_name} from tfimm")
            embed = pretrained_model(inputs[0])
            # create_model(nb_classes=0) includes pooling as last layer

        # Bottleneck(s)
        for p, out_channels in zip(cfg.dropout_ps, cfg.lin_ftrs):
            embed = tf.keras.layers.Dropout(p)(embed)
            embed = tf.keras.layers.Dense(out_channels)(embed)
            embed = BatchNorm(bn_type=cfg.bn_head)(embed) if cfg.head_bn else embed
        if cfg.bn_head and not cfg.lin_ftrs:
            embed = BatchNorm(bn_type=cfg.bn_head)(embed)  # does this help?

        # Output layer or Margin
        if cfg.arcface and inference:
            output = embed
        elif cfg.arcface:
            margin = get_margin(cfg)
            features = margin([embed, inputs[1]])
            output = tf.keras.layers.Softmax(dtype='float32', name='arc' if cfg.aux_loss else None)(features)
        else:
            assert cfg.n_classes, 'set cfg.n_classes in project or config file!'
            features = tf.keras.layers.Dense(cfg.n_classes)(embed)
            output = tf.keras.layers.Softmax(dtype='float32')(features)
        
        if cfg.aux_loss:
            assert cfg.n_aux_classes, 'set cfg.n_aux_classes in project or config file!'
            aux_features = tf.keras.layers.Dense(cfg.n_aux_classes)(embed)
            aux_output = tf.keras.layers.Softmax(dtype='float32', name='aux')(aux_features)

        # Outputs
        outputs = [output]
        if cfg.aux_loss and not inference: 
            outputs.append(aux_output)

        # Build model
        if (cfg.n_acc > 1) and cfg.use_custom_training_loop:
            model = ModelWithGrad(inputs=inputs, outputs=outputs)
        elif cfg.n_acc > 1:
            model = CustomTrainStep(cfg.n_acc, inputs=inputs, outputs=outputs)
        else:
            model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        if cfg.freeze_bn:
            print("freezing layer", model.layers[1].name)
            freeze_bn(model.layers[1])  # freeze only backbone BN
            #model.layers[1].layers[2].trainable = True  # unfreeze stem BN

        if cfg.use_custom_training_loop: return model

        optimizer = (
            tfa.optimizers.AdamW(weight_decay=cfg.wd, learning_rate=cfg.lr, 
                                 beta_1=cfg.betas[0], beta_2=cfg.betas[1]) if cfg.optimizer == 'AdamW' else
            tf.keras.optimizers.Adam(learning_rate=cfg.lr, 
                                     beta_1=cfg.betas[0], beta_2=cfg.betas[1]) if cfg.optimizer == 'Adam' else
            tf.keras.optimizers.SGD(learning_rate=cfg.lr, momentum=cfg.betas[0])
            )

        cfg.metrics = cfg.metrics or []

        if 'acc' in cfg.metrics:
            metrics_classes['acc'] = tf.keras.metrics.SparseCategoricalAccuracy(name='acc')
        if 'top5' in cfg.metrics:
            metrics_classes['top5'] = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5')
        if 'f1' in cfg.metrics:
            metrics_classes['f1'] = tfa.metrics.FBetaScore(beta=1, average='micro', name='F1')
        if 'f2' in cfg.metrics:
            metrics_classes['f2'] = tfa.metrics.FBetaScore(beta=2, average='micro', name='F2')
        if 'macro_f1' in cfg.metrics:
            metrics_classes['macro_f1'] = tfa.metrics.FBetaScore(beta=1, average='macro', name='macro_F1')
        if 'non_existing_metric' in cfg.metrics:
            metrics_classes['non_existing_metric'] = tf.keras.metrics.NoSuchClass(name='non_existing_metric')

        metrics = [metrics_classes[m] for m in cfg.metrics]

        model.compile(
            optimizer = optimizer,
            loss = 'sparse_categorical_crossentropy',
            loss_weights = (1 - cfg.aux_loss, cfg.aux_loss) if cfg.aux_loss else None,
            metrics = metrics)

        check_model_inputs(cfg, model)

        return model