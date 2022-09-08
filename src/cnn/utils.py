import argparse
from type import dataset 
from  torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from typing import List 
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Rotation Invariance CNN')

    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--learning_rate", default=0.003, type=int)
    parser.add_argument("--model", default="lenet5", type=str)

    args = parser.parse_args()
    return args

def get_dataset(train: bool) -> dataset:

    train_dataset = datasets.MNIST(root="data", train=train, transform= ToTensor(), download=True)

    train_input = np.array(train_dataset.data.view(-1, 1, 28, 28).float())
    train_target = np.array(train_dataset.targets)
    
    name = "MNIST_train" if train else "MNIST_test"

    net_dataset: dataset = { 
        "name": name, 
        "input": train_input, 
        "target": train_target
    }

    return net_dataset

def set_numpy_settings():
    np.random.seed(0)
    np.seterr(invalid="ignore", over = "ignore")


def plot_loss(average_loss: List[int]):
    fig, ax = plt.subplots()
    ax.plot(average_loss)
    fig.savefig(os.path.join(os.path.dirname(__file__), "average_loss.png"))
    plt.close()



def average(arr: np.ndarray) -> int:
    return np.average(arr[0:1000])
   

