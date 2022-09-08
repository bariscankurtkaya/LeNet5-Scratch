from utils import *
from type import *
from network import create_LeNet5_network, create_fc_network, train_network

args = parse_args()
EPOCH = args.epoch
LEARNING_RATE = args.learning_rate
MODEL = args.model
set_numpy_settings()

train: dataset = get_dataset(train = True)
test: dataset = get_dataset(train = False)

if MODEL == "lenet5":
    new_network: network = create_LeNet5_network()
elif MODEL == "fully":
    new_network: network = create_fc_network()


loss = train_network(train= train, test=test, network=new_network, epoch = EPOCH, learning_rate= LEARNING_RATE)

plot_loss(loss)