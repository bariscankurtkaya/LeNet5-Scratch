from  torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from network import forward_prop_conv, forward_prop_fc, backward_prop_fc, update_params_fc
from utils import parse_args
import time

np.random.seed(0)
np.seterr(invalid="ignore", over = "ignore")

train_dataset = datasets.MNIST(root="data", train=True, transform= ToTensor(), download=True)
test_dataset = datasets.MNIST(root="data", train=False, transform= ToTensor(), download= True)

args = parse_args()
EPOCH = args.epoch

train_input = train_dataset.data.view(-1, 1, 28, 28).float()
train_target = train_dataset.targets

test_input = test_dataset.data.view(-1, 1, 28, 28).float()
test_target = test_dataset.targets

conv_kernel1 = np.random.randn(6,1,5,5)
conv_bias1 = np.random.randn(6, 1)

conv_kernel2 = np.random.randn(16,6,5,5)
conv_bias2 = np.random.randn(16, 1)

fc_weight1 = (np.random.randn(120, 400)) * np.sqrt(1./400)
fc_bias1 = np.random.randn(120, 1)

fc_weight2 = (np.random.randn(84, 120)) * np.sqrt(1./120)
fc_bias2 = np.random.randn(84, 1)

fc_weight3 = (np.random.randn(10, 84)) * np.sqrt(1./84)
fc_bias3 = np.random.randn(10, 1)

print(train_input.shape)
print((train_target.cpu().detach().numpy())[:10])

current_image : np.ndarray = []


for epoch_index in range(EPOCH):
    conv_outputs = []

    #time_start
    time_start = time.time()

    print(epoch_index, "th epoch started in", EPOCH, "!")

    #Conv forward
    for i in range(len(train_input)):

        current_image = train_input[i][0]
        flatten_output, conv_output1, max_pool_output1, conv_output2, max_pool_output2 = forward_prop_conv(image = current_image, conv_kernel1 = conv_kernel1, conv_bias1 = conv_bias1, conv_kernel2 = conv_kernel2, conv_bias2 = conv_bias2)
        conv_outputs.append(flatten_output)

        if i % 10000 == 0 or i == 60000:
            print(i,"th conv iteration")

    print("forward_conv_finished!")
    conv_outputs = np.array(conv_outputs).T

    #FC forward
    results,Z3,A2,Z2,A1,Z1 = forward_prop_fc(images = conv_outputs, weight1 = fc_weight1, bias1 = fc_bias1, weight2 = fc_weight2, bias2 = fc_bias2, weight3 = fc_weight3, bias3 = fc_bias3)
    print("forward_fc_finished!")
    print(results.shape)

    print("results: ", np.argmax(results, axis=0)[:10])


    dW3, db3, dW2, db2, dW1, db1, dA0 = backward_prop_fc(images = conv_outputs, classes = train_target, A3= results, Z3=Z3,A2=A2,Z2=Z2,A1=A1,Z1=Z1, W1 = fc_weight1, W2 = fc_weight2, W3 = fc_weight3)

    fc_weight3, fc_bias3, fc_weight2, fc_bias2, fc_weight1, fc_bias1 = update_params_fc(fc_weight3, fc_bias3, fc_weight2, fc_bias2, fc_weight1, fc_bias1, dW3, db3, dW2, db2, dW1, db1, 0.1)

    time_stop = time.time()
    print("whole process spent", round(time_stop - time_start) / 60 , "mins")

print(np.argmax(results, axis = 0)[:10])