#!/usr/bin/env python3
import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    
    for i in range(len(layer_sizes)):
        if i == 0:
            prediction = create_layer(x, layer_sizes[0], activations[0])
        else:
            prediction = create_layer(prediction, layer_sizes[i], activations[i])
            
    return prediction