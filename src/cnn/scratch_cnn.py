import numpy as np
import pandas as pd
import matplotlib
from scipy import signal


class Convolutional():
    def __init__(self, input_shape, kernel_size, kernel_count):
        self.k_count = kernel_count
        self.i_shape = input_shape
        self.i_depth, self.i_height, self.i_width = input_shape
        self.out_shape = (kernel_count, (self.i_height - kernel_size + 1), (self.i_width - kernel_size + 1))
        self.k_shape = (kernel_count, self.i_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.k_shape)
        self.biases = np.random.randn(*self.out_shape)
