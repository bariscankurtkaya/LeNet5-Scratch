from typing import Literal, Dict, List, TypedDict
import numpy as np

IMG= np.ndarray
KERNEL= np.ndarray
SCALARS= np.ndarray

correlation_mode = Literal["full", "valid", "same"]
boundary = Literal["fill", "wrap", "symm"]

activation_funcs = Literal["leaky", "relu", "sigmoid", "softmax"]
pooling = Literal["max", "mean", None]

# Layers
class conv_layer(TypedDict):
    kernel: KERNEL
    bias: KERNEL
    activation: str
    pooling: pooling

class fc_layer(TypedDict):
    weight: SCALARS
    bias: SCALARS
    activation: str


class fc_layers(TypedDict):
    fc_layers: List[fc_layer]

class conv_layers(TypedDict):
    conv_layers: List[conv_layer]


# Network
class network(TypedDict):
    conv_layers: conv_layers
    fc_layers: fc_layers



# Caches
class conv_cache(TypedDict):
    conv_inputs: IMG
    last_output: IMG

class fc_cache(TypedDict):
    activation_outputs: IMG #A
    layer_outputs: IMG #Z

class forward_cache(TypedDict):
    conv_cache: conv_cache 
    fc_cache: fc_cache 
    loss: np.ndarray

class bconv_cache(TypedDict):
    kernel_derivs: KERNEL
    bias_derivs: KERNEL

class bfc_cache(TypedDict):
    weight_derivs: SCALARS
    bias_derivs: SCALARS
    last_deriv : SCALARS

class backward_cache(TypedDict):
    bfc_cache: bfc_cache 
    bconv_cache: bconv_cache 



# Dataset
class dataset(TypedDict):
    name: str
    input: IMG
    target : SCALARS







