from typing import Literal

correlation_mode = Literal["full", "valid", "same"]
boundary = Literal["fill", "wrap", "symm"]

conv_layer: dict = {
    "kernel": [],
    "bias": [],
    "activation": str
}

fc_layer: dict = {
    "weight": [],
    "bias": [],
    "activation": str
}

fc_layers: dict[fc_layer] = []
conv_layers: dict[conv_layer] = []

network: dict = {
    conv_layers: conv_layers,
    fc_layers: fc_layers
} 

forward_cache: dict = {

}

dataset: dict = {
    "name": str,
    "input": [],
    "target": []
}