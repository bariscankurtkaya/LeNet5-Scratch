from type import *
import numpy as np
from signal_func import *
from math_eq import *


# Forward Functions
def forward_prop(input: IMG, network: network) -> forward_cache:
    net_forward_cache : forward_cache = create_forward_cache()
    
    if network["name"] == "lenet5":
        net_forward_cache["conv_cache"] = forward_prop_conv(input=input, conv_layers=network["conv_layers"])
        net_forward_cache["fc_cache"] = forward_prop_fc(input = net_forward_cache["conv_cache"]["last_output"], fc_layers=network["fc_layers"])

    elif network["name"] == "fully":
        net_forward_cache["fc_cache"] = forward_prop_fc(input = input, fc_layers=network["fc_layers"])

    return net_forward_cache



def forward_prop_conv(input: IMG, conv_layers:List[conv_layer]) -> conv_cache:
    net_conv_cache : conv_cache = create_conv_cache()

    for i in range(len(conv_layers)):
        net_conv_cache["conv_inputs"].append(input)

        output: np.ndarray = cross_correlation(input, conv_layers[i])
        input = output

    flatten_output = output.flatten()
    net_conv_cache["last_output"] = np.array([flatten_output]).T

    return net_conv_cache



def forward_prop_fc(input: IMG, fc_layers:List[fc_layer]) -> fc_cache:
    net_fc_cache : fc_cache = create_fc_cache()
    for i in range(len(fc_layers)):
        output: np.ndarray = fc_layers[i]["weight"].dot(input) + fc_layers[i]["bias"]
        activated_output = activation(input = output, activation=fc_layers[i]["activation"])
        net_fc_cache["layer_outputs"].append(output)
        net_fc_cache["activation_outputs"].append(activated_output)

        input = activated_output

    return net_fc_cache



# Backward Functions

def backward_prop(forward_cache: forward_cache, network: network, true_labels, input: IMG, hyperparameter:hyperparameters) -> network:
    net_backward_cache: backward_cache = create_backward_cache()
    true_labels = one_hot(true_labels)

    if network["name"] == "fully":
        net_backward_cache["bfc_cache"] = backward_prop_fc(forward_cache= forward_cache, fc_layers= network["fc_layers"], true_labels= true_labels, input=input)

    if network["name"] == "lenet5":
        net_backward_cache["bfc_cache"] = backward_prop_fc(forward_cache= forward_cache, fc_layers= network["fc_layers"], true_labels= true_labels, input=forward_cache["conv_cache"]["last_output"])
        net_backward_cache["bconv_cache"] = backward_prop_conv(backward_cache=net_backward_cache,conv_cache=forward_cache["conv_cache"], conv_layers= network["conv_layers"])
    
    network = update_params(backward_cache= net_backward_cache, network=network, hyperparameter=hyperparameter)
    del net_backward_cache
    return network




def backward_prop_fc(forward_cache: forward_cache, fc_layers: List[fc_layer], true_labels: np.ndarray, input:IMG) -> bfc_cache:
    net_bfc_cache : bfc_cache = create_bfc_cache()
    for i in range(len(fc_layers)):
        if i == 0:
            dZ = forward_cache["fc_cache"]["activation_outputs"][-1] - true_labels
        else:
            dZ = fc_layers[-i]["weight"].T.dot(dZ) * activation(input = forward_cache["fc_cache"]["activation_outputs"][-i-1], activation=fc_layers[-i-1]["activation"], derivative=True)

        if i+1 < len(fc_layers):
            net_bfc_cache["weight_derivs"].append(1/len(true_labels[0]) * dZ.dot(forward_cache["fc_cache"]["activation_outputs"][-i-2].T))
        else:
            net_bfc_cache["weight_derivs"].append(1/len(true_labels[0]) * dZ.dot(input.T))

        
        net_bfc_cache["bias_derivs"].append(1/len(true_labels[0]) * np.sum(dZ))

    net_bfc_cache["last_deriv"] = dZ #dZ1

    return net_bfc_cache




def backward_prop_conv(backward_cache: backward_cache, conv_cache: conv_cache, conv_layers: List[conv_layer]) -> bconv_cache:
    net_bconv_cache: bconv_cache = create_bconv_cache()

    dZ0 = np.dot(backward_cache["bfc_cache"]["weight_derivs"][-1].T, (backward_cache["bfc_cache"]["last_deriv"])) * activation(input = conv_cache["last_output"], activation= conv_layers[-1]["activation"], derivative=True)
    
    conv_output_deriv = flatten_2_kernel(dZ0.T)[0]

    for n in range(len(conv_layers)):
        if conv_layers[-n-1]["pooling"] is not None:
            conv_output_deriv = upsampling(conv_output_deriv)
        kernel_deriv = np.zeros(conv_layers[-n-1]["kernel"].shape)
        
        input_size = round(len(conv_cache["conv_inputs"][-n-1][0]))
        input_channel = round(len(conv_cache["conv_inputs"][-n-1]))
        net_bconv_cache["input_derivs"].append(np.zeros((input_channel, input_size, input_size)))

        bias_deriv = np.array([np.mean(np.mean(conv_output_deriv, axis= 2), axis= 1)]).T # I think bias should use mean of conv output derivative

        for i in range(len(conv_layers[-n-1]["kernel"])):
            for j in range(len(conv_layers[-n-1]["kernel"][i])):
                kernel_deriv[i][j] = cross_corr_func(conv_cache["conv_inputs"][-n-1][j], conv_output_deriv[i])
                net_bconv_cache["input_derivs"][n][j] = convolution_func(conv_output_deriv[i], conv_layers[-n-1]["kernel"][i][j]) * activation(input = conv_cache["conv_inputs"][-n-1][j], activation= conv_layers[-n-1]["activation"], derivative=True)
            
        net_bconv_cache["kernel_derivs"].append(kernel_deriv)
        net_bconv_cache["bias_derivs"].append(bias_deriv)

        conv_output_deriv = net_bconv_cache["input_derivs"][n]
    return net_bconv_cache


def update_params(backward_cache: backward_cache, network:network, hyperparameter:hyperparameters) -> network:
    if network["name"] == "lenet5":
        for i in range(len(network["conv_layers"])):
            network["conv_layers"][i]["kernel"] = network["conv_layers"][i]["kernel"] - hyperparameter["learning_rate"] * backward_cache["bconv_cache"]["kernel_derivs"][-i-1]
            network["conv_layers"][i]["bias"] = network["conv_layers"][i]["bias"] - hyperparameter["learning_rate"] * backward_cache["bconv_cache"]["bias_derivs"][-i-1]
    
    for i in range(len(network["fc_layers"])):
        network["fc_layers"][i]["weight"] = network["fc_layers"][i]["weight"] - hyperparameter["learning_rate"] * backward_cache["bfc_cache"]["weight_derivs"][-i-1]
        network["fc_layers"][i]["bias"] = network["fc_layers"][i]["bias"] - hyperparameter["learning_rate"] * backward_cache["bfc_cache"]["bias_derivs"][-i-1]
    
    return network



# Layer creation functions
def create_conv_layer(kernel_count:int, kernel_channel:int, kernel_size:int, activation:activation_funcs, pooling:pooling) -> conv_layer:
    net_conv_layer: conv_layer = {
        "kernel": np.random.rand(kernel_count, kernel_channel, kernel_size, kernel_size) - 0-5,
        "bias": np.random.rand(kernel_count, 1) - 0.5,
        "activation": activation,
        "pooling": pooling
    }

    return net_conv_layer



def create_fc_layer(input_size, output_size, activation) -> fc_layer:
    net_fc_layer: fc_layer = {
        "weight": (np.random.rand(output_size, input_size) - 0.5) * np.sqrt(1./input_size),
        "bias": np.random.rand(output_size, 1),
        "activation": activation
    }

    return net_fc_layer




# Pooling layer functions
def max_pool(input:IMG, kernel_size: int = 2, stride: int = 2, padding: int = 0) -> IMG:
    max_pool_output = []
    for i in range(len(input)):
        max_pool_output.append(pool2d(input[i][0], kernel_size = kernel_size, stride = stride, padding = padding))
    
    return np.array(max_pool_output)

def mean_pool(input:IMG, kernel_size: int = 2, stride: int = 2, padding: int = 0) -> IMG:
    mean_pool_output = []
    for i in range(len(input)):
        mean_pool_output.append(pool2d(input[i][0], kernel_size = kernel_size, stride = stride, padding = padding, pool_mode="mean"))
    
    return np.array(mean_pool_output)






# Inner layer functions
def cross_correlation(input:IMG, conv_layer: conv_layer) -> IMG:
    conv_output = []
    for i in range(len(conv_layer["kernel"])):
        conv_output.append(cross_corr_func(input, conv_layer["kernel"][i]) + np.mean(conv_layer["bias"][i]))
    
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
    if not derivative:
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
    one_hot_Y = np.zeros((true_labels.size, 10))
    one_hot_Y[np.arange(true_labels.size), true_labels] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def binary_cross_entropy(true_label: np.ndarray, predictions: np.ndarray) -> int:
    predictions = np.clip(predictions, 1e-7, 1 - 1e-7)
    true_labels = one_hot(true_label)
    term_0 = (1-true_labels) * np.log(1-predictions + 1e-7)
    term_1 = true_labels * np.log(predictions + 1e-7)
    loss = -np.mean(term_0+term_1, axis=0)

    del true_labels, term_0, term_1
    return loss