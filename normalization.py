from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization


def get_normalization_classes():
    "Return a tuple of all supported normalization classes"
    normalization_classes = [
        BatchNormalization,
        tf.keras.layers.LayerNormalization]
    if hasattr(tf.keras.layers.experimental, 'SyncBatchNormalization'):
        normalization_classes.append(tf.keras.layers.experimental.SyncBatchNormalization)
    if hasattr(tf.keras.layers, 'GroupNormalization'):
        normalization_classes.append(tf.keras.layers.GroupNormalization)
    return tuple(normalization_classes)


def Normalization(layer_class, name=None, gn_groups=None, vbs=None):
    """Return an instance of a keras normalization layer

    `layer_class [str | tf.keras.layers.Layer]: keyword, layer class name, or layer class
    Default: BatchNormalization
    Keywords: 'BN', 'GN', 'SyncBN', 'layer_norm', 'inctance_norm'
    """
    if isinstance(layer_class, tf.keras.layers.Layer):
        return layer_class(name=name)
    if layer_class is True:
        return BatchNormalization(name=name)  # default
    if not isinstance(layer_class, str):
        raise TypeError(f"layer_class must be a str or keras Layer class, got {type(layer_class)}")
    if layer_class.lower() in {'bn', 'batch_norm'}:
        return BatchNormalization(name=name)
    if layer_class.lower() in {'syncbn', 'sync_bn'}:
        return BatchNormalization(name=name, synchronized=True)
    if layer_class.lower() == 'gn':
        return tf.keras.layers.GroupNormalization(name=name, groups=gn_groups or 1)
    if layer_class == 'layer_norm':
        return tf.keras.layers.LayerNormalization(name=name)  # bad valid, nan loss
    if layer_class == 'instance_norm':
        return BatchNormalization(virtual_batch_size=vbs, name=name)
    if hasattr(tf.keras.layers, layer_class):
        return getattr(tf.keras.layers, layer_class)(name=name)
    raise ValueError(f'{layer_class} is not a recognized normalization class')


def replace_bn_layers(model, layer_class, keep_weights=False, n_gpu=1, **kwargs):
    """Replace all BatchNormalization layers in `model`
    layer_class must be supported by Normalization
    Derived from:
    https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
    """
    vbs = kwargs.get('vbs', None)
    if isinstance(layer_class, str) and (layer_class.lower() == 'syncbn') and n_gpu < 2:
        print("Info: SyncBN only affects distributed GPU training!")
    n_replacements = 0

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer._outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update({layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # All but output layers must have outbound nodes
    if not network_dict['input_layers_of']:
        print(f"replace_bn_layers: model {model.name} has no layer connectivity, cannot reconstruct.")
        print("WARNING: keeping model with original normalization layers.")
        return model

    # Set the output tensor of the input layer
    input = (model.input if hasattr(model, 'input') else 
             model.inputs[0] if model.inputs else 
             tf.keras.layers.Input(shape=(64, 64, 3), name='image'))
    if hasattr(model, 'inputs') and hasattr(model.inputs, 'len') and len(model.inputs) > 1:
        print("WARNING: model has several inputs, replace_bn_layers untested for this case")
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: input})

    # Iterate over all layers after the input
    model_outputs = []
    for layer in model.layers[1:]:

        # Determine input tensors
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                       for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Replace layer (keep name)
        if isinstance(layer, BatchNormalization):
            n_replacements += 1
            new_layer = Normalization(layer_class, **kwargs, name=layer.name)
            if keep_weights:
                new_layer.build(layer.input_shape)
                for w_old, w_new in zip(layer.weights, new_layer.weights):
                    if vbs:
                        # virtual batch size adds extra dims: (1, 1, 1, 1, dim)
                        w_old = tf.reshape(w_old, w_new.shape)
                    w_new.assign(w_old)
            x = new_layer(layer_input)
            #kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            #print(f'replaced {layer.name} by {x.__class__.__name__}({kwargs_str})')
        elif layer.name.startswith('tf.math.reduce_mean'):
            # call with inferred kwargs
            axis = [i for i, n in enumerate(layer.get_output_shape_at(0)) if n == 1]
            x = layer(layer_input, axis=axis, keepdims=True)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

        # Save tensor in output list if it is output in initial model
        if layer.name in model.output_names:
            model_outputs.append(x)

    inputs = model.inputs
    new_model = Model(inputs=inputs, outputs=model_outputs, name=model.name)

    # Now all nodes are duplicated.
    # Save and load model to clean up Graph (recommended on StackOverflow)
    try:
        new_model.save('tmp.keras')
        new_model = tf.keras.models.load_model('tmp.keras')
    except ValueError as e:
        if 'axis' in str(e):
            # Work around BatchNormalization/VBS bug, axis must be 3 during save, else 4
            for l in new_model.layers:
                if isinstance(l, tf.keras.layers.BatchNormalization) and l.virtual_batch_size is not None:
                    #print("axis before save:", l.axis[0])  # 4
                    if hasattr(l.axis, '__len__'):
                        l.axis[0] -= 1
                    else:
                        l.axis -= 1

            new_model.save('tmp.keras')
            new_model = tf.keras.models.load_model('tmp.keras')

            for l in new_model.layers:
                if isinstance(l, tf.keras.layers.BatchNormalization) and l.virtual_batch_size is not None:
                    #print("axis after load:", l.axis[0])  # 4 (!)
                    if hasattr(l.axis, '__len__') and l.axis[0] == 3:
                        l.axis[0] += 1
                    elif l.axis == 3:
                        l.axis += 1
                    #print("restored axis:", l.axis[0])
        else:
            raise
    try:
        Path('tmp.keras').unlink()
    except FileNotFoundError:
        pass  # kwarg "missing_ok" introduced in python 3.8

    if n_replacements:
        print(f"Replaced {n_replacements} instances of BatchNormalization with"
              f" {new_layer.__class__.__name__}{'(virtual_batch_size)' if vbs else ''}.")

    return new_model
