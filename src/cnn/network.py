from signal_func import prepare_img_to_LeNet5
from type import *
from layer import create_conv_layer, create_fc_layer,forward_prop, backward_prop, binary_cross_entropy



def create_LeNet5_network() -> network:
    conv1: conv_layer = create_conv_layer(kernel_count=6, kernel_channel=1, kernel_size=5, activation="leaky", pooling="max")
    conv2: conv_layer = create_conv_layer(kernel_count=16, kernel_channel=6, kernel_size=5, activation="leaky", pooling="max")

    fc1: fc_layer = create_fc_layer(input_size=400, output_size=120, activation="leaky")
    fc2: fc_layer = create_fc_layer(input_size=120, output_size=84, activation="leaky")
    fc3: fc_layer = create_fc_layer(input_size=84, output_size=10, activation="softmax")

    conv_layers: conv_layers = [conv1, conv2]
    fc_layers: fc_layers = [fc1, fc2, fc3]

    lenet5: network = {
        "conv_layers": conv_layers,
        "fc_layers": fc_layers
    }

    return lenet5



def use_LeNet5(train: dataset, test:dataset, lenet5: network, epoch: int, learning_rate: int):
    for i in range(epoch):
        for i in range(len(train["input"])):
            img: IMG = prepare_img_to_LeNet5(input = train["input"][i])

            forward_cache: forward_cache = forward_prop(input=img, network=lenet5)

            forward_cache["loss"]:np.ndarray = binary_cross_entropy(true_labels= train["target"][i], predictions=forward_cache["fc_cache"]["activation_outputs"][-1])

            print(forward_cache["loss"])

            lenet5 = backward_prop(forward_cache= forward_cache, network=lenet5, true_labels= train["target"][i], learning_rate= learning_rate)









