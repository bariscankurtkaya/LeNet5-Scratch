from turtle import down
from  torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from network import convolution, max_pool, softmax, Leaky_ReLU, Leaky_ReLU_deriv

np.random.seed(0)

train_dataset = datasets.MNIST(root="data", train=True, transform= ToTensor(), download=True)
test_dataset = datasets.MNIST(root="data", train=False, transform= ToTensor(), download= True)


train_input = train_dataset.data.view(-1, 1, 28, 28).float()
train_target = train_dataset.targets
test_input = test_dataset.data.view(-1, 1, 28, 28).float()
test_target = test_dataset.targets

conv_kernel1 = np.random.rand(6,1,5,5) - 0.5
conv_bias1 = np.random.rand(6, 1)

conv_kernel2 = np.random.rand(16,6,5,5) - 0.5
conv_bias2 = np.random.rand(16, 1)

fc_weight1 = np.random.rand(120, 400)
fc_bias1 = np.random.rand(120)

fc_weight2 = np.random.rand(84, 120)
fc_bias2 = np.random.rand(84)

#print(train_input.shape)
#print(train_target.shape)


current_image : np.ndarray = []

#for i in range(len(train_input)):
    
current_image = train_input[1][0]
padded_image = np.array([np.pad(current_image, 2, mode='constant')])

print("current_image:", current_image[1][0].shape)
print("padded_image:",padded_image.shape)

conv_output1 = convolution(padded_image, conv_kernel1, conv_bias1)
print("conv_output1:",conv_output1.shape)

max_pool_output1 = max_pool(conv_output1, kernel_size = 2, stride = 2, padding = 0 )
print("max_pool_output1:", max_pool_output1.shape)

conv_output2 = convolution(max_pool_output1, conv_kernel2, conv_bias2)
print("conv_output2:", conv_output2.shape)

max_pool_output2 = max_pool(conv_output2, kernel_size = 2, stride = 2, padding = 0 )
print("max_pool_output2:",max_pool_output2.shape)

flatten_output = max_pool_output2.flatten().reshape((400,1))
print(flatten_output.shape)


    
