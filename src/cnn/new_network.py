from signal_func import *
import numpy as np
from math_eq import * 
from type import *

def cross_correlation(input, conv_kernel, conv_bias) -> np.ndarray:
    conv_output = []
    for i in range(len(conv_kernel)):
        conv_output.append(cross_corr_func(input, conv_kernel[i]) + conv_bias[i]) 

    return np.array(conv_output)

def max_pool(input, kernel_size, stride, padding) -> np.ndarray:
    max_pool_output = []
    for i in range(len(input)):
        max_pool_output.append(pool2d(input[i][0], kernel_size = kernel_size, stride = stride, padding = padding, pool_mode = "max"))
    
    return np.array(max_pool_output)


def create_LeNet5_network():
    
