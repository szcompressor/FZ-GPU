import numpy as np
import torch
import ctypes
from ctypes import *
from random import random
from itertools import accumulate

# load so library to python
pfz = ctypes.CDLL('./fz-gpu.so', mode=ctypes.RTLD_GLOBAL)

# initialize the input and output
input_tensor = torch.tensor([1 for i in range(1024 * 1024)], dtype=torch.float32).cuda()
output_tensor = torch.tensor([0 for i in range(1024 * 1024)], dtype=torch.int16).cuda()

input_ptr = cast(input_tensor.data_ptr(), POINTER(c_float))
output_ptr = cast(output_tensor.data_ptr(), POINTER(c_uint16))

inputSize = c_int(1024 * 1024)
errorBound = c_float(0.1)
x = c_int(1024)
y = c_int(1024)
z = c_int(1)
times = c_int(10)

pfz.pyTestQuantization(input_ptr,
                    #    output_ptr,
                       inputSize,
                       errorBound,
                       x,
                       y,
                       z,
                       times)
