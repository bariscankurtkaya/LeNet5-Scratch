from  torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from network import forward_prop_conv, forward_prop_fc
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

conv_kernel1 = np.random.rand(6,1,5,5) - 0.5
conv_bias1 = np.random.rand(6, 1)

conv_kernel2 = np.random.rand(16,6,5,5) - 0.5
conv_bias2 = np.random.rand(16, 1)

fc_weight1 = np.random.rand(120, 400) - 0.5
fc_bias1 = np.random.rand(120, 1)

fc_weight2 = np.random.rand(84, 120) - 0.5
fc_bias2 = np.random.rand(84, 1)

fc_weight3 = np.random.rand(10, 84) - 0.5
fc_bias3 = np.random.rand(10, 1)

print(train_input.shape)
#print(train_target.shape)



current_image : np.ndarray = []


for epoch_index in range(EPOCH):
    conv_outputs = []

    #time_start
    time_start = time.time()

    print(epoch_index, "th epoch started in", EPOCH, "!")

    #Conv forward
    for i in range(len(train_input)):

        current_image = train_input[i][0]
        flatten_output = forward_prop_conv(image = current_image, conv_kernel1 = conv_kernel1, conv_bias1 = conv_bias1, conv_kernel2 = conv_kernel2, conv_bias2 = conv_bias2)
        conv_outputs.append(flatten_output)

        if i % 10000 == 0 or i == 60000:
            print(i,"th iteration")

    print("forward_conv_finished!")
    conv_outputs = np.array(conv_outputs).T

    #FC forward
    results = forward_prop_fc(images = conv_outputs, weight1 = fc_weight1, bias1 = fc_bias1, weight2 = fc_weight2, bias2 = fc_bias2, weight3 = fc_weight3, bias3 = fc_bias3)
    print("forward_fc_finished!")
    print(results.shape)


    time_stop = time.time()
    print("whole process costs", round(time_stop - time_start) / 60 , "mins")