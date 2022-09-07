from type import *
import numpy as np
from signal_func import *
from math_eq import *



# Forward Functions
def forward_prop(input: IMG, network: network) -> forward_cache:
    forward_cache : forward_cache

    forward_cache["conv_cache"] = forward_prop_conv(input=input, conv_layers=network["conv_layers"])

    forward_cache["fc_cache"] = forward_prop_fc(input = forward_cache["conv_cache"]["last_output"], fc_layers=network["fc_layers"])

    return forward_cache



def forward_prop_conv(input: IMG, conv_layers:conv_layers) -> conv_cache:
    conv_cache : conv_cache
    for i in range(len(conv_layers)):
        conv_cache["conv_inputs"][i] = input

        output: np.ndarray = cross_correlation(input, conv_layers[i])
        input = output

    conv_cache["last_output"] = output.flatten()

    return conv_cache



def forward_prop_fc(input: IMG, fc_layers:fc_layers) -> fc_cache:
    fc_cache : fc_cache
    for i in range(len(fc_layers)):
        output: np.ndarray = fc_layers[i]["weight"].dot(input) + fc_layers[i]["bias"]
        activated_output = activation(input = output, activation=fc_layers[i]["activation"])

        fc_cache["layer_outputs"][i] = output
        fc_cache["activation_outputs"][i] = activated_output

    return fc_cache



# Backward Functions

def backward_prop(forward_cache: forward_cache, network: network, true_labels):
    backward_cache: backward_cache
    true_labels = one_hot(true_labels)

    backward_cache["bfc_cache"] = backward_prop_fc(forward_cache= forward_cache, fc_layers= network["fc_layers"], true_labels= true_labels)
    
    return




def backward_prop_fc(forward_cache: forward_cache, fc_layers: fc_layers, true_labels= np.ndarray) -> bfc_cache:
    bfc_cache : bfc_cache
    for i in range(len(fc_layers)):
        if i == 0:
            dZ = forward_cache["fc_cache"]["activation_outputs"][-1] - true_labels
        else:
            dZ = fc_layers[-i].T.dot(dZ) * activation(input = forward_cache["fc_cache"]["activation_outputs"][-1], activation=fc_layers[-i-1]["activation"], derivative=True)

        if i+1 < len(fc_layers):
            bfc_cache["weight"][i] = 1/len(true_labels[0]) * dZ.dot(forward_cache["fc_cache"]["activation_outputs"][-i-2])
        else:
            bfc_cache["weight"][i] = 1/len(true_labels[0]) * dZ.dot(forward_cache["conv_cache"]["last_output"])
        
        bfc_cache["bias"][i] = 1/len(true_labels[0]) * np.sum(dZ)

    bfc_cache["last_deriv"] = dZ

    return bfc_cache




def backward_prop_conv():
    return




# Layer creation functions
def create_conv_layer(kernel_count:int, kernel_channel:int, kernel_size:int, activation:activation_funcs, pooling:pooling) -> conv_layer:
    conv_layer: conv_layer = {
        "kernel": np.random.rand(kernel_count, kernel_channel, kernel_size, kernel_size),
        "bias": np.random.rand(kernel_count, kernel_size, kernel_size),
        "activation": activation,
        "pooling": pooling
    }

    return conv_layer



def create_fc_layer(input_size, output_size, activation) -> fc_layer:
    fc_layer: fc_layer = {
        "weight": (np.random.rand(output_size, input_size) - 0.5) * np.sqrt(1./input_size),
        "bias": np.random.rand(output_size, 1),
        "activation": activation
    }

    return fc_layer




# Pooling layer functions
def max_pool(input:IMG, kernel_size: int, stride: int = 2, padding: int = 0) -> IMG:
    max_pool_output = []
    for i in range(len(input)):
        max_pool_output.append(pool2d(input[i][0], kernel_size = kernel_size, stride = stride, padding = padding))
    
    return np.array(max_pool_output)

def mean_pool(input:IMG, kernel_size: int, stride: int = 2, padding: int = 0) -> IMG:
    mean_pool_output = []
    for i in range(len(input)):
        mean_pool_output.append(pool2d(input[i][0], kernel_size = kernel_size, stride = stride, padding = padding, pool_mode="mean"))
    
    return np.array(mean_pool_output)






# Inner layer functions
def cross_correlation(input:IMG, conv_layer: conv_layer) -> IMG:
    conv_output = []
    for i in range(len(conv_layer["kernel"])):
        conv_output.append(cross_corr_func(input, conv_layer["kernel"][i]) + conv_layer["bias"][i])
    
    conv_output = np.array(conv_output)
    conv_output = activation(input=conv_output, activation=conv_layer["activation"])

    if conv_layer["pooling"] == "max":
        return max_pool(conv_output)
    elif conv_layer["pooling"] == "mean":
        return mean_pool(conv_output)
    elif conv_layer["pooling"] == None:
        return conv_output
    else:
        raise Exception("There is no pooling function as", conv_layer["pooling"])


def activation(input: IMG, activation: activation_funcs, derivative: bool = False) -> IMG:
    if derivative:
        if activation == "leaky":
            return Leaky_ReLU(input)
        elif activation == "relu":
            return ReLU(input)
        elif activation == "sigmoid":
            return sigmoid(input)
        elif activation == "softmax":
            return softmax(input)
        else:
            raise Exception("There is no activation function as", activation)
    else:
        if activation == "leaky":
            return Leaky_ReLU_deriv(input)
        elif activation == "relu":
            return ReLU_deriv(input)
        elif activation == "sigmoid":
            return sigmoid_deriv(input)
        else:
            raise Exception("There is no activation function deriv as", activation)


# Loss functions

def one_hot(true_labels) -> np.ndarray:
    Y = true_labels.cpu().detach().numpy()
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def binary_cross_entropy(true_labels: np.ndarray, predictions: np.ndarray) -> int:
    true_labels = true_labels.cpu().detach().numpy()
    predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
    term_0 = (1-true_labels) * np.log(1-predictions + 1e-7)
    term_1 = true_labels * np.log(predictions + 1e-7)
    return -np.mean(term_0+term_1, axis=0)
