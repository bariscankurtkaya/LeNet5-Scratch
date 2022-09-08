from signal_func import prepare_img_to_LeNet5, prepare_to_fc_network
from type import *
from layer import create_conv_layer, create_fc_layer,forward_prop, backward_prop, binary_cross_entropy
from utils import save_avg_and_delete


def create_LeNet5_network() -> network:
    conv1: conv_layer = create_conv_layer(kernel_count=6, kernel_channel=1, kernel_size=5, activation="leaky", pooling="max")
    conv2: conv_layer = create_conv_layer(kernel_count=16, kernel_channel=6, kernel_size=5, activation="leaky", pooling="max")

    #fc1: fc_layer = create_fc_layer(input_size=400, output_size=120, activation="leaky")
    fc1: fc_layer = create_fc_layer(input_size=1024, output_size=120, activation="leaky")
    fc2: fc_layer = create_fc_layer(input_size=120, output_size=84, activation="leaky")
    fc3: fc_layer = create_fc_layer(input_size=84, output_size=10, activation="softmax")

    net_conv_layers: List[conv_layer] = [conv1, conv2]
    net_fc_layers: List[fc_layer] = [fc1, fc2, fc3]

    lenet5: network = {
        "name": "lenet5",
        "conv_layers": net_conv_layers,
        "fc_layers": net_fc_layers
    }

    return lenet5


def create_fc_network() -> network:

    fc1: fc_layer = create_fc_layer(input_size=1024, output_size=120, activation="leaky")
    fc2: fc_layer = create_fc_layer(input_size=120, output_size=84, activation="leaky")
    fc3: fc_layer = create_fc_layer(input_size=84, output_size=10, activation="softmax")

    net_fc_layers: List[fc_layer] = [fc1, fc2, fc3]

    fc_network: network = {
        "name": "fully",
        "fc_layers": net_fc_layers
    }

    return fc_network

def train_network(train: dataset, test:dataset, network: network) -> List[int]:
    for n in range(EPOCH):
        print(f'{n}th epoch started in {EPOCH}!')
        for i in range(len(train["input"])):
            
            if i % SAVE_COUNT == 0:
                print(f'{i}th iteration started in {len(train["input"])}!')
                if i != 0:
                    save_avg_and_delete()

            
            if network["name"] == "lenet5":
                img: IMG = prepare_img_to_LeNet5(img = train["input"][i])
            if network["name"] == "fully":
                img: IMG = prepare_to_fc_network(img = train["input"][i])

            net_forward_cache: forward_cache = forward_prop(input=img, network=network)

            LOSS.append(binary_cross_entropy(true_label= train["target"][i], predictions=net_forward_cache["fc_cache"]["activation_outputs"][-1]))

            network = backward_prop(forward_cache= net_forward_cache, network=network, true_labels= train["target"][i], input=img)
            del net_forward_cache

    return network, LOSS_average






