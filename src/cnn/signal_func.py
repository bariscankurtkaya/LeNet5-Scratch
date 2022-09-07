from scipy import signal
import numpy as np
from numpy.lib.stride_tricks import as_strided
from type import *


def cross_corr_func(img: IMG, kernel: np.ndarray, mode="valid") -> np.ndarray:
    return signal.correlate(img, kernel, mode)

def convolution_func(img: IMG, kernel: np.ndarray, mode="full") -> np.ndarray:
    return signal.convolve(img, kernel, mode)

def pool2d(img:IMG, kernel_size: int, stride:int = 2, padding:int = 0, pool_mode:str='max') -> np.ndarray:
    """
    2D Pooling

    Parameters:
        img: input 2D array
        kernel_size: int, the size of the window over which we take pool
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    """
    # Padding
    img = padding_func(img, padding=padding)

    # Window view of img
    output_shape = ((img.shape[0] - kernel_size) // stride + 1,
                    (img.shape[1] - kernel_size) // stride + 1)
    
    shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
    strides_w = (stride*img.strides[0], stride*img.strides[1], img.strides[0], img.strides[1])
    
    img_w = as_strided(img, shape_w, strides_w)

    # Return the result of pooling
    if pool_mode == 'max':
        return img_w.max(axis=(2, 3))
    elif pool_mode == 'avg':
        return img_w.mean(axis=(2, 3))

def padding_func(img: IMG, padding:int = 2, mode:str = "constant") ->IMG:
    return np.pad(img, padding, mode=mode)


def prepare_img_to_LeNet5(input:IMG) -> IMG:
    input = input / 255
    input = padding_func(np.array(input))

    return input

def flatten_2_kernel(x):
    x = np.reshape(x, (len(x),16,5,5))
    return x


def upsampling(x, upkernel_size: int = 2):
    x = x.repeat(upkernel_size, axis=1).repeat(upkernel_size, axis=2)
    return x