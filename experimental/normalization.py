import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization


def replace_bn_layers(model, layer_class, keep_weights=False, **kwargs):
    """Derived from:
    https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model"""
    class_name = layer_class.__name__
    kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    vbs = kwargs.get('virtual_batch_size', None)
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

    # Set the output tensor of the input layer
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

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
            new_layer = layer_class(**kwargs, name=layer.name)
            if keep_weights:
                new_layer.build(layer.input_shape)
                for w_old, w_new in zip(layer.weights, new_layer.weights):
                    if vbs:
                        # virtual batch size adds extra dims: (1, 1, 1, 1, dim)
                        w_old = tf.reshape(w_old, w_new.shape)
                    w_new.assign(w_old)
            x = new_layer(layer_input)
            #print(f'replaced {layer.name} by {class_name}({kwargs_str})')
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
        new_model.save('tmp.h5')
        new_model = tf.keras.models.load_model('tmp.h5')
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

            new_model.save('tmp.h5')
            new_model = tf.keras.models.load_model('tmp.h5')

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

    if n_replacements:
        print(f"Replaced {n_replacements} instances of BatchNormalization by"
              f" {class_name}{'(virtual_batch_size)' if vbs else ''}.")

    return new_model