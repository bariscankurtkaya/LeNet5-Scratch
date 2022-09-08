from utils import *
from type import hyperparameters, parse_args, set_hyperparameter
from network import create_LeNet5_network, create_fc_network, train_network

args = parse_args()
hyperparameter: hyperparameters = set_hyperparameter(args)
MODEL = args.model

set_numpy_settings()

train: dataset = get_dataset(train = True)
test: dataset = get_dataset(train = False)

if MODEL == "lenet5":
    new_network: network = create_LeNet5_network()
elif MODEL == "fully":
    new_network: network = create_fc_network()


new_network, loss = train_network(train= train, test=test, network=new_network, hyperparameter=hyperparameter)

plot_loss(loss)
save_model(new_network)