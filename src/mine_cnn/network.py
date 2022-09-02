from signal_func import cross_correlation_2d, pool2d
import numpy as np

def convolution(input, conv_kernel, conv_bias):
    conv_output = []
    for i in range(len(conv_kernel)):
        conv_output.append(cross_correlation_2d(input, conv_kernel[i]) + conv_bias[i]) 

    return conv_output


def max_pool(input, kernel_size, stride, padding):
    max_pool_output = []
    for i in range(len(input)):
        max_pool_output.append(pool2d(input[i], kernel_size = kernel_size, stride = stride, padding = padding, pool_mode = "max"))
    
    return max_pool_output

def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A