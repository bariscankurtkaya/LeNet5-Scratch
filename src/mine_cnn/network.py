from signal_func import cross_correlation, pool2d
import numpy as np

def convolution(input, conv_kernel, conv_bias):
    #print("input:", input.shape, "conv: ", conv_kernel.shape)
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


def forward_prop_conv(image, conv_kernel1, conv_bias1, conv_kernel2, conv_bias2) -> np.ndarray:
    padded_image = np.array([np.pad(image, 2, mode='constant')])

    #print("current_image:", image.shape)
    #print("padded_image:",padded_image.shape)

    conv_output1 = convolution(padded_image, conv_kernel1, conv_bias1)
    #print("conv_output1:",conv_output1.shape)

    max_pool_output1 = max_pool(conv_output1, kernel_size = 2, stride = 2, padding = 0 )
    #print("max_pool_output1:", max_pool_output1.shape)

    conv_output2 = convolution(max_pool_output1, conv_kernel2, conv_bias2)
    #print("conv_output2:", conv_output2.shape)

    max_pool_output2 = max_pool(conv_output2, kernel_size = 2, stride = 2, padding = 0 )
    #print("max_pool_output2:",max_pool_output2.shape)

    flatten_output = max_pool_output2.flatten()
    #print(flatten_output.shape)

    return flatten_output

def forward_prop_fc(images, weight1, bias1, weight2, bias2, weight3, bias3):
    Z1 = weight1.dot(images) + bias1
    A1 = Leaky_ReLU(Z1)
    Z2 = weight2.dot(A1) + bias2
    A2 = Leaky_ReLU(Z2)
    Z3 = weight3.dot(A2) + bias3
    A3 = softmax(Z3)

    return A3