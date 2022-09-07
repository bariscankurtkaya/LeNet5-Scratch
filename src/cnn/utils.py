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


def plot_loss(loss: List[int]):

    average_loss = average(loss)
    plt.plot(loss)
    plt.savefig(os.path.join(os.path.dirname(__file__), "loss.png"))

    plt.plot(average_loss)
    plt.savefig(os.path.join(os.path.dirname(__file__), "average_loss.png"))


def average(arr):
  average_arr = []
  for i in range(int(len(arr)/100)):
      average_arr.append(np.average(arr[i*100:(i+1)*100]))
  return average_arr

