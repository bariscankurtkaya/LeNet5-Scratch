from scipy import signal
from type import correlation_mode, boundary
import numpy as np

def cross_correlation_2d(matrix: np.ndarray, kernel: np.ndarray, boundary: boundary = "fill", mode: correlation_mode ="valid", fillvalue: int = 0) -> np.ndarray:
    return signal.correlate2d(matrix, kernel, boundary = boundary, mode = mode, fillvalue = fillvalue)

