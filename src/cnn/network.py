from signal_func import cross_corr_func, pool2d, convolution_func
import numpy as np

def cross_correlation(input, conv_kernel, conv_bias):
    #print("input:", input.shape, "conv: ", conv_kernel.shape)
    conv_output = []
    for i in range(len(conv_kernel)):
        conv_output.append(cross_corr_func(input, conv_kernel[i]) + conv_bias[i]) 

    return np.array(conv_output)


def max_pool(input, kernel_size, stride, padding):
    max_pool_output = []
    for i in range(len(input)):
        max_pool_output.append(pool2d(input[i][0], kernel_size = kernel_size, stride = stride, padding = padding, pool_mode = "max"))
    
    return np.array(max_pool_output)



def forward_prop_conv(image, conv_kernel1, conv_bias1, conv_kernel2, conv_bias2) -> np.ndarray:
    padded_image = np.array([np.pad(image, 2, mode='constant')])

    #print("current_image:", image.shape)
    #print("padded_image:",padded_image.shape)

    conv_output1 = convolution(padded_image, conv_kernel1, conv_bias1)
    #print("conv_output1:",conv_output1.shape)

    relu_conv_output1 = Leaky_ReLU(conv_output1)

    max_pool_output1 = max_pool(relu_conv_output1, kernel_size = 2, stride = 2, padding = 0 )
    #print("max_pool_output1:", max_pool_output1.shape)

    conv_output2 = convolution(max_pool_output1, conv_kernel2, conv_bias2)
    #print("conv_output2:", conv_output2.shape)

    relu_conv_output2 = Leaky_ReLU(conv_output2)

    max_pool_output2 = max_pool(relu_conv_output2, kernel_size = 2, stride = 2, padding = 0 )
    #print("max_pool_output2:",max_pool_output2.shape)

    flatten_output = max_pool_output2.flatten()

    #print(flatten_output.shape)

    return flatten_output, relu_conv_output1, max_pool_output1, relu_conv_output2, max_pool_output2, padded_image


def forward_prop_fc(images, weight1, bias1, weight2, bias2, weight3, bias3):
    Z1 = weight1.dot(images) + bias1
    A1 = Leaky_ReLU(Z1)
    Z2 = weight2.dot(A1) + bias2
    A2 = Leaky_ReLU(Z2)
    Z3 = weight3.dot(A2) + bias3
    A3 = softmax(Z3)

    #print("weight1: ", np.max(weight1), "\nweight2: ",  np.max(weight2), "\nweight3: ",  np.max(weight3))

    return A3,Z3,A2,Z2,A1,Z1


def backward_prop_fc(images, classes, A3, Z3, A2, Z2, A1, Z1, W1, W2, W3):
    train_image_count = len(images[0])
    one_hot_classes = one_hot(classes)

    dZ3 = A3 - one_hot_classes
    dW3 = (1/train_image_count) * dZ3.dot(A2.T)
    db3 = (1/train_image_count) * np.sum(dZ3)

    dZ2 = W3.T.dot(dZ3) * Leaky_ReLU_deriv(Z2)
    dW2 = (1/train_image_count) * dZ2.dot(A1.T)
    db2 = (1/train_image_count) * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * Leaky_ReLU_deriv(Z1)
    dW1 = (1/train_image_count) * dZ1.dot(images.T)
    db1 = (1/train_image_count) * np.sum(dZ1)

    dZ0 = W1.T.dot(dZ1) * Leaky_ReLU_deriv(images)

    #print("dZ1 shape", dZ1.shape, "dZ2 shape", dZ2.shape, "dZ3 shape", dZ3.shape)

    return dW3, db3, dW2, db2, dW1, db1, dZ0


def update_params_fc(W3, b3, W2, b2, W1, b1, dW3, db3, dW2, db2, dW1, db1, alpha):
        
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2

        W3 = W3 - alpha * dW3
        b3 = b3 - alpha * db3

        #print("dw1: ", np.max(dW1), "\ndW2: ",  np.max(dW2), "\ndW3: ",  np.max(dW3))


        return W3, b3, W2, b2, W1, b1


def one_hot(Y):
    Y = Y.cpu().detach().numpy()
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop_conv(input1, input2, dZ0, conv_kernel1, conv_bias1, conv_kernel2, conv_bias2, alpha):
    #CK -> Conv Kernel
    #CB -> Conv Bias
    # dZ0 = dY2
    print(dZ0.shape)
    dZ0 = flatten_2_kernel(dZ0.T)
    print(dZ0.shape)

    for n in range(len(dZ0)):
        currentdA0 = upsampling(dZ0[n])
        #print(currentdA0.shape)

        dCK2 = np.zeros((16,6,5,5))
        dY1 = np.zeros((6,14,14))

        dCK1 = np.zeros((6,1,5,5))

        #print(dA0[0].shape, conv_kernel2[0].shape)
        for i in range(len(conv_kernel2)):
            for j in range(len(conv_kernel2[i])):
                dCK2[i][j] = cross_correlation(input2[j], currentdA0[i])
                dY1[j] = convolution_func(currentdA0[i], conv_kernel2[i][j])

        #print(dCK2.shape)

        dB2 = currentdA0

        dY1 = dY1 * Leaky_ReLU_deriv(input2)

        dY1 = upsampling(dY1)


        dB1 = dY1

        for i in range(len(conv_kernel1)):
            for j in range(len(conv_kernel1[i])):
                dCK1[i][j] = cross_correlation(input1[j], dY1[i])

        conv_kernel2 = np.subtract(conv_kernel2, alpha*dCK2)
        conv_kernel1 = np.subtract(conv_kernel1, alpha*dCK1)

        dB2 = np.average(dB2, axis=2)
        dB2 = np.average(dB2, axis=1)
        #print(dB2.shape)
        conv_bias2 = conv_bias2 - alpha * dB2

        conv_bias2 = np.resize(conv_bias2, (16,1))

        dB1 = np.average(dB1, axis=2)
        dB1 = np.average(dB1, axis=1)
        conv_bias1 = conv_bias1 - alpha * dB1 

        conv_bias1 = np.resize(conv_bias1, (6,1))

    print("dck2: ", dCK2[1], "\ndck1: ", dCK1[1])
    return conv_kernel1, conv_bias1, conv_kernel2, conv_bias2


def flatten_2_kernel(x):
    x = np.reshape(x, (len(x),16,5,5))
    return x


def upsampling(x):
    x = x.repeat(2, axis=1).repeat(2, axis=2)
    return x