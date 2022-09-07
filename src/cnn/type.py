from typing import Literal, List, TypedDict
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


# Network
class network(TypedDict):
    conv_layers: List[conv_layer]
    fc_layers: List[fc_layer]



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
    loss: int

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




def create_conv_cache() -> conv_cache:
    new_conv_cache: conv_cache = {
        "conv_inputs": [],
        "last_output": np.array([])
    }
    return new_conv_cache

def create_fc_cache() -> fc_cache:
    new_fc_cache: fc_cache = {
        "activation_outputs": [],
        "layer_outputs": []
    }
    return new_fc_cache


def create_forward_cache() -> forward_cache:
    new_forward_cache: forward_cache = {
        "conv_cache" : create_conv_cache(),
        "fc_cache" : create_fc_cache(),
        "loss" : 0
    }
    return new_forward_cache