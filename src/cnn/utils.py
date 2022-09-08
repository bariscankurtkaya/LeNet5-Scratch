from type import dataset, LOSS_AVERAGE, LOSS, MIN_LOSS, network, CURRENT_LOSS
from  torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from typing import List 
import os
import csv


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
    global LOSS, CURRENT_LOSS, LOSS_AVERAGE

    CURRENT_LOSS = average(LOSS)
    LOSS_AVERAGE.append(CURRENT_LOSS)
    plot_loss(LOSS_AVERAGE)
    LOSS.clear()


def check_best_model():
    global MIN_LOSS

    if MIN_LOSS > CURRENT_LOSS :
        print("MIN LOSS: ", MIN_LOSS)
        MIN_LOSS = CURRENT_LOSS


def save_model(network: network):
    # open file for writing, "w" is writing
    w = csv.writer(open("model.csv", "w"))

    # loop over dictionary keys and values
    for key, val in network.items():

        # write every key and value to file
        w.writerow([key, val])
