from type import *
import numpy as np


def create_conv_layer(kernel_count, kernel_channel, kernel_size) -> conv_layer:
    conv_layer: conv_layer = {
        "kernel": np.random.rand(kernel_count, kernel_channel, kernel_size, kernel_size),
        "bias": np.random.rand(kernel_count, kernel_size, kernel_size)
    }

    return conv_layer

def create_fc_layer(input_size, output_size) -> fc_layer:
    fc_layer: fc_layer = {
        "weight": (np.random.rand(output_size, input_size) - 0.5) * np.sqrt(1./input_size),
        "bias": np.random.rand(output_size, 1)
    }

    return fc_layer