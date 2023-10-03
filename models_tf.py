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
- swin_base_patch4_window7_224_in22k OK in CPU/GPU, error on TPU (XLA cannot infer shape
    for model/swin_transformer_1/layers/0/blocks/0/attn/qkv/Tensordot operation)
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
    - UnimplementedError:  Fused conv implementation does not support grouped convolutions.
"""
import math
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import (Input, Flatten, Dense, Dropout, Softmax,
                                     GlobalAveragePooling2D, GlobalMaxPooling2D, concatenate)
from normalization import Normalization, get_normalization_classes

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
        inputs = tf.pow(inputs, 1.0 / self.p)
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


class GammaCurve(tf.keras.layers.Layer):
    def __init__(self, predict_inverse=True, **kwargs):
        super().__init__(**kwargs)
        self.predict_inverse = predict_inverse

    def call(self, bp, bp2, log_gamma):
        support = tf.linspace(0.0, 1.0, 256)
        x = support[None, None, :]
        bp = bp[:, :, None]
        bp2 = bp2[:, :, None]
        gamma = tf.exp(log_gamma)[:, :, None]
        if self.predict_inverse:
            x = bp + x * (1 - bp)
            x = tf.clip_by_value(x, 1e-6, 1)  # avoid nan
            x = tf.pow(x, gamma)
            x = x * (1 - bp2) + bp2
            return tf.clip_by_value(x, 0, 1)
        else:
            x = (x - bp2) / (1 - bp2)
            x = tf.clip_by_value(x, 1e-6, 1)  # avoid nan
            x = tf.pow(x, 1 / gamma)
            x = (x - bp) / (1 - bp)
            return tf.clip_by_value(x, 0, 1)


def get_margin(cfg):
    # Adaptive margins for each target class (range: cfg.margin_min ... cfg.margin_max)
    # should be defined in project.
    m = cfg.adaptive_margin or 0.3

    if cfg.arcface == 'ArcMarginProduct':
        return ArcMarginProductSubCenter(cfg.n_classes, m=m, k=cfg.subcenters or 1,
                                         easy_margin=cfg.easy_margin,
                                         name=f'head/{cfg.arcface}', dtype='float32')

    if cfg.arcface == 'AddMarginProduct':
        return AddMarginProductSubCenter(cfg.n_classes, m=m, k=cfg.subcenters or 1,
                                         name=f'head/{cfg.arcface}', dtype='float32')

    raise ValueError(f'ArcFace type {cfg.arcface} not supported')


def check_model_inputs(cfg, model):
    for inp in model.inputs:
        assert inp.name in cfg.data_format, f'"{inp.name}" (model.inputs) missing in cfg.data_format'


def freeze_bn(model, unfreeze=False):
    "Freeze or (unfreeze=True) unfreeze all normalization layers"
    normalization_classes = get_normalization_classes()
    for layer in model.layers:
        if isinstance(layer, normalization_classes):
            layer.trainable = unfreeze
            #print(f"setting {layer.name} trainable to {unfreeze}")


def set_trainable(model, freeze):
    """Set trainable attributes according to list `freeze`
    
    "none":       all layers trainable (default)
    "all":        all layers frozen
    "head":       freeze all head layers
    "body":       freeze all body layers
    "bn":         freeze all normalization layers in the body
    "all_but_bn": only normalization layers are trainable
    "preprocess": freeze any preprocess layer
    """
    freeze = freeze or set(['none'])
    freeze = set(s.lower() for s in freeze)

    if 'none' in freeze:
        model.trainable = True
        return
    
    if 'all' in freeze:
        print("freezing entire model")
        model.trainable = False
        return
    
    # Freeze only specified parts of the model
    # Notes: 
    # - layers are only trained if also their parents are trainable
    # - moving_mean, moving_var in BN are always non-trainable params but updated 
    #   in training if (layer.momentum < 1) and (layer.trainable is True)
    model.trainable = True
    body_index = 2 if model.layers[1].name.endswith('transform_tf') else 1
    first_head_layer_index = body_index + 1

    if 'head' in freeze:
        print("freezing head layers")
        for layer in model.layers[first_head_layer_index:]:
            layer.trainable = False

    if 'body' in freeze:
        body = model.layers[body_index]
        print("freezing body:", body.name)
        body.trainable = False

    if 'bn' in freeze:
        # freeze only BN layers in body
        body = model.layers[body_index]
        print("freezing normalization layers in", body.name)
        freeze_bn(body)
        #body.layers[2].trainable = True  # unfreeze stem BN

    if 'all_but_bn' in freeze:
        print("freezing all except normalization layers")
        for layer in model.layers:
            layer.trainable = False
        freeze_bn(model, unfreeze=True)  # BN in head
        body = model.layers[body_index]
        body.trainable = True
        for layer in body.layers:
            layer.trainable = False
        freeze_bn(body, unfreeze=True)  # BN in body

    if 'preprocess' in freeze and model.layers[1].name.endswith('transform_tf'):
        print("freezing preprocess layer:", model.layers[1].name)
        model.layers[1].trainable = False


def set_bn_parameters(model, momentum=None, eps=None, debug=False):
    "Modify `momentum` and `eps` (epsilon) parameters in all supported normalization layers"
    if not (momentum or eps): return
    body_index = 2 if model.layers[1].name.endswith('transform_tf') else 1
    body = model.layers[body_index]

    normalization_classes = get_normalization_classes()

    n_replaced = 0
    for layer in body.layers:
        if isinstance(layer, normalization_classes):
            if momentum:
                if debug and layer.momentum != momentum:
                    print(f"{layer.name}: changing momentum {layer.momentum} -> {momentum}")
                layer.momentum = momentum
            if eps:
                if debug and layer.epsilon != eps:
                    print(f"{layer.name}: changing eps {layer.epsilon} -> {eps}")
                layer.epsilon = eps
    if n_replaced:
        print(f"Set momentum, epsilon in {n_replaced} normalization layers")


def check_model_weights(model):
    names = set()
    valid = True

    for w in model.weights:
        if w.name in names:
            valid = False
            print(f'WARNING: found duplicate weight names:')
            for i, w_i in enumerate(model.weights):
                if w_i.name == w.name:
                    print(f"    {i:<3} {w_i.name}")
        names.add(w.name)

    if not valid: 
        raise ValueError("Duplicate weight names: this model will not save/load correctly!")


def get_bottleneck_params(cfg):
    "Define one Dropout (maybe zero) per FC + optional final Dropout"
    dropout_ps = cfg.dropout_ps or []
    lin_ftrs = cfg.lin_ftrs or []
    if len(dropout_ps) > len(lin_ftrs) + 1:
        raise ValueError(f"too many dropout_ps ({len(dropout_ps)}) for {len(lin_ftrs)} lin_ftrs")
    final_dropout = dropout_ps.pop() if len(dropout_ps) == len(lin_ftrs) + 1 else 0
    num_missing_ps = len(lin_ftrs) - len(dropout_ps)
    dropout_ps.extend([0] * num_missing_ps)

    return lin_ftrs, dropout_ps, final_dropout


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
        import tensorflow_hub as hub
        print("tensorflow_hub:", hub.__version__)
    else:
        import tfimm
        print("tfimm:", tfimm.__version__)
        import timm
        print("timm:", timm.__version__)
        if cfg.list_models:
            print(tfimm.list_models(pretrained="timm"))

    with strategy.scope():

        # Inputs
        input_shape = (*cfg.size, 3)
        inputs = [Input(shape=input_shape, name='image')]
        if cfg.arcface and not inference:
            inputs.append(Input(shape=(), name='target'))

        # Preprocessing
        x = cfg.preprocess()(inputs[0]) if hasattr(cfg.preprocess, 'call') else inputs[0]

        # Body
        efnv1 = cfg.arch_name.startswith('efnv1')
        efnv2 = cfg.arch_name.startswith('efnv2')
        tfhub = cfg.arch_name in TFHUB

        pretrained_model = (
            model_cls(weights=cfg.pretrained, input_shape=input_shape, include_top=False) if efnv1 else
            model_cls(input_shape=input_shape, num_classes=0, pretrained=cfg.pretrained) if efnv2 else
            hub.KerasLayer(TFHUB[cfg.arch_name], trainable=True) if tfhub else
            tfimm.create_model(cfg.arch_name, pretrained=True, nb_classes=0, input_size=cfg.size))

        if cfg.normalization:
            from normalization import replace_bn_layers
            pretrained_model = replace_bn_layers(pretrained_model, cfg.normalization, 
                                                 keep_weights=True,
                                                 gn_groups=cfg.gn_groups,
                                                 vbs=cfg.virtual_batch_size,
                                                 n_gpu=cfg.gpu)

        x = pretrained_model(x)

        # Head(s)
        if efnv1:
            if isinstance(cfg.pool, tf.keras.layers.Layer):
                embed = cfg.pool(x, inputs) if hasattr(cfg.pool, 'requires_inputs') else cfg.pool(x)
            elif cfg.pool == 'flatten':
                embed = Flatten()(x)
            elif cfg.pool == 'fc':
                embed = Flatten()(x)
                embed = Dropout(0.1)(embed)
                embed = Dense(1024)(embed)
            elif cfg.pool == 'gem':
                embed = GeMPoolingLayer(train_p=True)(x)
            elif cfg.pool == 'concat':
                embed = concatenate([GlobalAveragePooling2D()(x),
                                                     GlobalAveragePooling2D()(x)])
            elif cfg.pool == 'max':
                embed = GlobalMaxPooling2D()(x)
            else:
                embed = GlobalAveragePooling2D()(x)

        elif efnv2:
            if isinstance(cfg.pool, tf.keras.layers.Layer):
                print(f"pool requires inputs:", hasattr(cfg.pool, 'requires_inputs'))
                print(f"calling pool with x {x.shape}, inputs [{inputs[0].shape}]")
                embed = cfg.pool(x, inputs) if hasattr(cfg.pool, 'requires_inputs') else cfg.pool(x)
            elif cfg.pool == 'flatten':
                embed = Flatten()(x)
            elif cfg.pool == 'fc':
                embed = Flatten()(x)
                embed = Dropout(0.1)(embed)
                embed = Dense(1024)(embed)
            elif cfg.pool == 'gem':
                embed = GeMPoolingLayer(train_p=True)(x)
            elif cfg.pool == 'concat':
                embed = concatenate([GlobalAveragePooling2D()(x),
                                                     GlobalAveragePooling2D()(x)])
            elif cfg.pool == 'max':
                embed = GlobalMaxPooling2D()(x)
            else:
                embed = GlobalAveragePooling2D()(x)

        elif tfhub:
            # tfhub models cannot be modified => Pooling cannot be changed!
            assert cfg.pool in [None, False, 'avg', ''], 'tfhub model, no custom pooling supported!'
            print(f"{cfg.arch_name} from tfhub")
            embed = x

        else:
            print(f"{cfg.arch_name} from tfimm")
            embed = x
            # create_model(nb_classes=0) includes pooling as last layer

        # Bottleneck(s)
        if cfg.get_bottleneck is not None:
            for layer in cfg.get_bottleneck(cfg):
                embed = layer(embed)
        else:
            lin_ftrs, dropout_ps, final_dropout = get_bottleneck_params(cfg)
            for i, (p, out_channels) in enumerate(zip(dropout_ps, lin_ftrs)):
                embed = Dropout(p, name=f"dropout_{i}_{p}")(embed) if p > 0 else embed
                embed = Dense(out_channels, activation=cfg.act_head, name=f"FC_{i}")(embed)
                embed = Normalization(cfg.normalization_head, name=f"BN_{i}")(embed) if cfg.normalization_head else embed
            embed = Dropout(final_dropout, name=f"dropout_final_{final_dropout}")(
                embed) if final_dropout else embed
            if cfg.normalization_head and not lin_ftrs:
                embed = Normalization(cfg.normalization_head, name="BN_final")(embed)  # does this help?

        # Output layer or Margin
        if cfg.arcface and inference:
            output = embed
        elif cfg.arcface:
            margin = get_margin(cfg)
            features = margin([embed, inputs[1]])
            output = Softmax(dtype='float32', name='arc' if cfg.aux_loss else None)(features)
        elif cfg.classes or cfg.n_classes:
            assert cfg.n_classes, 'set cfg.n_classes in project or config file!'
            features = Dense(cfg.n_classes, name='classifier')(embed)
            output = Softmax(dtype='float32')(features)
        else:
            assert cfg.channel_size, 'set cfg.channel_size in project or config file!'
            if cfg.curve and (cfg.curve == 'gamma'):
                out_bp = Dense(cfg.channel_size, name='bp')(embed)
                out_bp2 = Dense(cfg.channel_size, name='bp2')(embed)
                out_log_gamma = Dense(cfg.channel_size, name='log_gamma')(embed)

                # calculate out_cuve(out_bp, out_bp2, out_log_gamma) -> 0...1 [B, 3, 255]
                if False:
                    support = tf.linspace(0.0, 1.0, 256)
                    x = support[None, None, :]
                    bp = out_bp[:, :, None]
                    bp2 = out_bp2[:, :, None]
                    gamma = tf.exp(out_log_gamma)[:, :, None]
                    if cfg.predict_inverse:
                        x = bp + x * (1 - bp)
                        x = tf.clip_by_value(x, 1e-6, 1)  # avoid nan
                        x = tf.pow(x, gamma)
                        x = x * (1 - bp2) + bp2
                        out_curve = tf.clip_by_value(x, 0, 1, name='curve')
                    else:
                        x = (x - bp2) / (1 - bp2)
                        x = tf.clip_by_value(x, 1e-6, 1)  # avoid nan
                        x = tf.pow(x, 1 / gamma)
                        x = (x - bp) / (1 - bp)
                        out_curve = tf.clip_by_value(x, 0, 1, name='curve')
                else:
                    out_curve = GammaCurve(predict_inverse=cfg.predict_inverse,
                                           name='curve')(out_bp, out_bp2, out_log_gamma)
            elif cfg.curve and (cfg.channel_size == 6):
                out_gamma = Dense(3, name='regressor_gamma')(embed)
                out_bp = Dense(3, name='regressor_bp')(embed)
            elif cfg.curve and (cfg.channel_size == 9):
                out_a = Dense(3, name='regressor_a')(embed)
                out_b = Dense(3, name='regressor_b')(embed)
                out_bp = Dense(3, name='regressor_bp')(embed)
            else:
                output = Dense(cfg.channel_size, name='regressor')(embed)

        if cfg.aux_loss:
            assert cfg.n_aux_classes, 'set cfg.n_aux_classes in project or config file!'
            aux_features = Dense(cfg.n_aux_classes, name='aux_classifier')(embed)
            aux_output = Softmax(dtype='float32', name='aux')(aux_features)

        # Outputs
        if cfg.curve and (cfg.curve == 'gamma'):
            outputs = ([out_curve, out_bp, out_bp2, out_log_gamma] if cfg.output_curve_params else
                       [out_curve])
        elif cfg.curve and (cfg.channel_size == 6):
            # slicing does not work, output shapes are still [?, 6]
            #outputs = [output[:3], output[3:]]  # gamma, bp
            outputs = [out_gamma, out_bp]
        elif cfg.curve and (cfg.channel_size == 9):
            outputs = [out_a, out_b, out_bp]
        elif cfg.curve and (cfg.curve == 'free'):
            # We don't care about curve outside the valid range
            #outputs = [tf.clip_by_value(output, 0, 1, name='clipped_curve')]
            outputs = [output]
        else:
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

        # Load restart weights
        if cfg.rst_path and cfg.rst_name:
            # avoid ValueError "axes don't match array":
            # https://stackoverflow.com/questions/51944836/keras-load-model-valueerror-axes-dont-match-array
            set_trainable(model, cfg.freeze_for_loading)

            try:
                model.load_weights(Path(cfg.rst_path) / cfg.rst_name)
            except ValueError:
                print(f"{cfg.rst_name} mismatches model with body: {model.layers[1].name}")
                print("Trying to load matching layers only...")
                model.load_weights(Path(cfg.rst_path) / cfg.rst_name, 
                                   by_name=True, skip_mismatch=True)
            print(f"Weights loaded from {cfg.rst_name}")

        # Freeze/unfreeze, set BN parameters
        set_trainable(model, cfg.freeze)
        set_bn_parameters(model, momentum=cfg.bn_momentum, eps=cfg.bn_eps, debug=cfg.DEBUG)

        # The following can probably be moved out of the strategy.scope...

        if cfg.use_custom_training_loop: return model

        optimizer = (
            tf.keras.optimizers.AdamW(learning_rate=cfg.lr, weight_decay=cfg.wd,
                                      beta_1=cfg.betas[0],
                                      beta_2=cfg.betas[1]) if cfg.optimizer == 'AdamW' else
            tf.keras.optimizers.Adam(learning_rate=cfg.lr,
                                     beta_1=cfg.betas[0],
                                     beta_2=cfg.betas[1]) if cfg.optimizer == 'Adam' else
            tf.keras.optimizers.SGD(learning_rate=cfg.lr, momentum=cfg.betas[0]))

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

        if cfg.aux_loss:
            loss_weights=(1 - cfg.aux_loss, cfg.aux_loss)
        elif cfg.curve and (cfg.curve == 'gamma'):
            loss_weights = (1, 0, 0, 0) if cfg.output_curve_params else None  # curve, bp, bp2, log_gamma
        elif cfg.curve and (cfg.channel_size == 6):
            # gamma, bp
            loss_weights = cfg.loss_weights or (1, 0.01 ** 2)  # error goal: 0.01, 1
            assert len(loss_weights) == len(outputs)
        elif cfg.curve and (cfg.channel_size == 9):
            # a, b, bp
            loss_weights = cfg.loss_weights or (1, 1, 1)
            assert len(loss_weights) == len(outputs)
        else:
            loss_weights = None
        if loss_weights:
            print("Loss weights:", *loss_weights)

        steps_per_execution = cfg.steps_per_execution or 1

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy' if cfg.classes else 'mean_squared_error',
            loss_weights=loss_weights,
            metrics=metrics,
            steps_per_execution=steps_per_execution)

    if steps_per_execution > 1:
        print(f"Model compiled with steps_per_execution={steps_per_execution}")
        if not cfg.validation_steps:
            print("Warning: validation will fail with steps_per_execution > 1 if validation_steps cannot be inferred.")

    check_model_inputs(cfg, model)
    check_model_weights(model)

    return model
