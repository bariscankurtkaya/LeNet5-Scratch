from utils import *

set_numpy_settings()

train: dataset = get_dataset(train = True)
test: dataset = get_dataset(train = False)