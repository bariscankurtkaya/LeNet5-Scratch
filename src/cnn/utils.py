import argparse
from type import dataset 
from  torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Rotation Invariance CNN')

    parser.add_argument("--epoch", default=10, type=int)

    args = parser.parse_args()
    return args

def get_dataset(train: bool) -> dataset:

    train_dataset = datasets.MNIST(root="data", train=train, transform= ToTensor(), download=True)

    train_input = np.array(train_dataset.data.view(-1, 1, 28, 28).float())
    train_target = np.array(train_dataset.targets)
    
    name = "MNIST_train" if train else "MNIST_test"

    dataset: dataset = { 
        "name": name, 
        "input": train_input, 
        "target": train_target
    }

    return dataset

def set_numpy_settings():
    np.random.seed(0)
    np.seterr(invalid="ignore", over = "ignore")