import argparse
from type import dataset, LOSS_average, LOSS, MIN_LOSS, network, CURRENT_LOSS
from  torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from typing import List 
import os
import csv


def parse_args():
    parser = argparse.ArgumentParser(description='Rotation Invariance CNN')

    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--learning_rate", default=0.003, type=int)
    parser.add_argument("--model", default="lenet5", type=str)
    parser.add_argument("--save_count", default=1000, type=int)


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
   
def save_avg_and_delete():
    CURRENT_LOSS = average(LOSS)
    LOSS_average.append(CURRENT_LOSS)
    plot_loss(LOSS_average)
    LOSS.clear()


def check_best_model(network: network):
    if MIN_LOSS > CURRENT_LOSS and (MIN_LOSS / CURRENT_LOSS) > 5:
        save_model(network)


def save_model(network: network):
    # open file for writing, "w" is writing
    w = csv.writer(open("model.csv", "w"))

    # loop over dictionary keys and values
    for key, val in network.items():

        # write every key and value to file
        w.writerow([key, val])
