from signal_func import cross_correlation, pool2d
import numpy as np

def convolution(input, conv_kernel, conv_bias):
    print("input:", input.shape, "conv: ", conv_kernel.shape)
    conv_output = []
    for i in range(len(conv_kernel)):
        conv_output.append(cross_correlation(input, conv_kernel[i]) + conv_bias[i]) 

    return np.array(conv_output)


def max_pool(input, kernel_size, stride, padding):
    max_pool_output = []
    for i in range(len(input)):
        max_pool_output.append(pool2d(input[i][0], kernel_size = kernel_size, stride = stride, padding = padding, pool_mode = "max"))
    
    return np.array(max_pool_output)

def softmax(Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

def Leaky_ReLU(Z):
        Z = np.where(Z > 0, Z, Z * 0.1)
        return Z
    
def Leaky_ReLU_deriv(Z):
        Z = np.where(Z > 0, 1, 0.1)
        return Z