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




def create_type(class_type: type):
    if class_type == conv_layer:
        return {
            "kernel": KERNEL,
            "bias": KERNEL,
            "activation": str,
            "pooling": pooling
        }
        
    if class_type == fc_layer:
        return {}
        weight: SCALARS
        bias: SCALARS
        activation: str

    # Network
    if class_type == network:
        return {}
        conv_layers: conv_layers
        fc_layers: fc_layers



    # Caches
    if class_type == conv_cache:
        return {}
        conv_inputs: IMG
        last_output: IMG

    if class_type == fc_cache:
        return {}
        activation_outputs: IMG #A
        layer_outputs: IMG #Z

    if class_type == forward_cache:
        return {}
        conv_cache: conv_cache 
        fc_cache: fc_cache 
        loss: np.ndarray

    if class_type == bconv_cache:
        return {}
        kernel_derivs: KERNEL
        bias_derivs: KERNEL

    if class_type == bfc_cache:
        return {}
        weight_derivs: SCALARS
        bias_derivs: SCALARS
        last_deriv : SCALARS

    if class_type == backward_cache:
        return {}
        bfc_cache: bfc_cache 
        bconv_cache: bconv_cache 



    # Dataset
    if class_type == dataset:
        return {}
        name: str
        input: IMG
        target : SCALARS
    

