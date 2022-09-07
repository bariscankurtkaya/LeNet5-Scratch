from typing import Literal
import numpy as np

IMG: np.ndarray
KERNEL: np.ndarray
SCALARS: np.ndarray

correlation_mode = Literal["full", "valid", "same"]
boundary = Literal["fill", "wrap", "symm"]

activation_funcs = Literal["leaky", "relu", "sigmoid", "softmax"]
pooling = Literal["max", "mean", None]

# Layers
conv_layer: dict = {
    "kernel": KERNEL,
    "bias": KERNEL,
    "activation": str,
    "pooling": Literal["max", "mean", None]
}

fc_layer: dict = {
    "weight": SCALARS,
    "bias": SCALARS,
    "activation": str
}

fc_layers: dict[fc_layer]
conv_layers: dict[conv_layer]



# Network
network: dict = {
    "conv_layers": conv_layers,
    "fc_layers": fc_layers
} 



# Caches
conv_cache: dict = {
    "conv_inputs" : IMG,
    "last_output" : IMG
}

fc_cache: dict = {
    "activation_outputs": IMG, #A
    "layer_outputs" : IMG #Z
}

forward_cache: dict = {
    "conv_cache" : conv_cache,
    "fc_cache" : fc_cache,
    "loss": np.ndarray
}


bfc_cache: dict = {
    "weight_derivs": SCALARS,
    "bias_derivs" : SCALARS,
    "last_deriv" : SCALARS
}

bconv_cache: dict = {
    "kernel_derivs": KERNEL,
    "bias_derivs": KERNEL

}


backward_cache: dict = {
    "bfc_cache" : bfc_cache,
    "bconv_cache" : bconv_cache
}


# Dataset
dataset: dict = {
    "name": str,
    "input": np.ndarray,
    "target": np.ndarray
}





