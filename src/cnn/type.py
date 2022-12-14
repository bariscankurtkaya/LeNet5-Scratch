from typing import Literal, List, Type, TypedDict
import numpy as np
import argparse

IMG= np.ndarray
KERNEL= np.ndarray
SCALARS= np.ndarray

MIN_LOSS: int = 100
CURRENT_LOSS: int = 100

LOSS: List = []
LOSS_AVERAGE: List = []



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
    name: str
    conv_layers: List[conv_layer]
    fc_layers: List[fc_layer]


class hyperparameters(TypedDict):
    epoch: int
    learning_rate: int
    save_count: int

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
    input_derivs: KERNEL

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

def create_bfc_cache() -> bfc_cache:
    new_bfc_cache: bfc_cache = {
        "weight_derivs": [],
        "bias_derivs": [],
        "last_deriv": np.array([])
    }
    return new_bfc_cache

def create_bconv_cache() -> bconv_cache:
    new_bconv_cache: bconv_cache = {
        "kernel_derivs": [],
        "bias_derivs": [],
        "input_derivs": []
    }
    return new_bconv_cache

def create_backward_cache() -> backward_cache:
    new_backward_cache: backward_cache = {
        "bfc_cache": create_bfc_cache,
        "bconv_cache": create_bconv_cache
    }
    return new_backward_cache


def parse_args():
    parser = argparse.ArgumentParser(description='Rotation Invariance CNN')

    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--learning_rate", default=0.003, type=float)
    parser.add_argument("--model", default="lenet5", type=str)
    parser.add_argument("--save_count", default=1000, type=int)


    args = parser.parse_args()
    return args

def set_hyperparameter(args):
    hyperparameter: hyperparameters = {
        "epoch": args.epoch,
        "learning_rate": args.learning_rate,
        "save_count" : args.save_count
    }
    return hyperparameter