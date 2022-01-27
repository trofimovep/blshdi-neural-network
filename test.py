import numpy as np
from numpy import ndarray

arr: ndarray = np.random.rand(2)
print(arr)

arr.put(0, 13)
print(arr)