from utils import *
from type import *
from network import create_LeNet5_network, use_LeNet5

args = parse_args()
EPOCH = args.epoch
LEARNING_RATE = args.learning_rate

set_numpy_settings()

train: dataset = get_dataset(train = True)
test: dataset = get_dataset(train = False)

lenet5: network = create_LeNet5_network()

loss = use_LeNet5(train= train, test=test, lenet5=lenet5, epoch = EPOCH, learning_rate= LEARNING_RATE)

plot_loss(loss)