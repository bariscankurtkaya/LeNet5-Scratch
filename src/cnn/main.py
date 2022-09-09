from utils import *
from type import hyperparameters, parse_args, set_hyperparameter
from network import create_LeNet5_network, create_fc_network, train_network, test_network

args = parse_args()
hyperparameter: hyperparameters = set_hyperparameter(args)
MODEL = args.model
READ_MODEL = args.read_model

set_numpy_settings()

train: dataset = get_dataset(train = True)
test: dataset = get_dataset(train = False)

if not READ_MODEL:
    if MODEL == "lenet5":
        new_network: network = create_LeNet5_network()
    elif MODEL == "fully":
        new_network: network = create_fc_network()


    new_network, loss = train_network(train= train, network=new_network, hyperparameter=hyperparameter)

    plot_loss(loss)
    save_model(new_network)

else:
    new_network = set_model()

test_result = test_network(test = test, network = new_network)

print("%",test_result)

