import numpy as np
import torch
import ctypes
from ctypes import *
from random import random

# create example tensors on GPU
input_tensor_gpu0 = torch.tensor([1 for i in range(1024 * 1024)], dtype=torch.float32).cuda()
input_tensor_gpu1 = torch.tensor([2 for i in range(1024 * 1024)], dtype=torch.float32).cuda()
input_list = [cast(input_tensor_gpu0.data_ptr(), POINTER(c_float)), cast(input_tensor_gpu1.data_ptr(), POINTER(c_float))]
input_list_c = (POINTER(c_float) * 2)(*input_list)
input_list_prt = pointer(input_list_c)

gpu_index = c_int(0)

input_size = c_int(1024 * 1024)

input_list_size = c_int(2)

world_size = c_int(4)

error_bound = c_float(1)

dimension_info = (c_int*3)(*[1024, 256, 1])
dimension_info_ptr = pointer(dimension_info)

compressed_tensor_gpu = torch.tensor([3 for i in range(1024 * 1024 * 4 * 2)], dtype = torch.uint8).cuda()
compressed_ptr = cast(compressed_tensor_gpu.data_ptr(), POINTER(c_uint8))

compressed_size = [[0, 0, 0, 0], [0, 0, 0, 0]]
# Create a 2D array
compressed_size_c = ((c_int * 4) * 2)()
# Assign values to the 2D array
for i in range(2):
    for j in range(4):
        compressed_size_c[i][j] = compressed_size[i][j]
compressed_size_ptr = pointer(compressed_size_c)

# load so library to python
pfz = ctypes.CDLL('./fz-gpu.so', mode=ctypes.RTLD_GLOBAL)

# launch compression
pfz.pfzCompress(input_list_prt,
                gpu_index,
                input_size,
                input_list_size,
                world_size,
                error_bound,
                dimension_info_ptr,
                compressed_ptr,
                compressed_size_ptr
                )

# launch decompression
# pfz.pzfDecompress()