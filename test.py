import numpy as np
import torch
import ctypes
from ctypes import *
from random import random
from itertools import accumulate

# load so library to python
pfz = ctypes.CDLL('./fz-gpu.so', mode=ctypes.RTLD_GLOBAL)

# create tensors and pointers for compression
input_tensor_gpu0 = torch.tensor([1 for i in range(1024 * 1024)], dtype=torch.float32).cuda()
input_tensor_gpu1 = torch.tensor([1 for i in range(1024 * 1024)], dtype=torch.float32).cuda()
input_list = [cast(input_tensor_gpu0.data_ptr(), POINTER(c_float)), cast(input_tensor_gpu1.data_ptr(), POINTER(c_float))]
input_list_c = (POINTER(c_float) * 2)(*input_list)
input_list_prt = pointer(input_list_c)

gpu_index = c_int(0)

input_size = c_int(1024 * 1024)

input_list_size = c_int(2)

world_size = c_int(4)

error_bound = c_float(0.1)

dimension_info = (c_int*3)(*[1024 * 256, 1, 1])
dimension_info_ptr = pointer(dimension_info)

compressed_tensor_gpu = torch.tensor([3 for i in range(1024 * 1024 * 4 * 2)], dtype = torch.uint8).cuda()
compressed_ptr = cast(compressed_tensor_gpu.data_ptr(), POINTER(c_uint8))

compressed_size_c = (c_int * 8)()
compressed_size_ptr = pointer(compressed_size_c)

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

# create tensors and pointers for decompression
compressed_size = [s for s in compressed_size_c]
offset_list = list(accumulate(compressed_size))
offset_list = [0, ] + offset_list[:-1]
offset_list_c = (c_int*8)(*offset_list)
print(offset_list)
offset_list_ptr = pointer(offset_list_c)

offset_list_size = c_int(8)

gpu_index = 0

decompressed_tensor_list = [torch.tensor([0 for i in range(1024 * 256)], dtype=torch.float32).cuda() for i in range(8)]
decompressed_tensor_ptr_list = [cast(decompressed_tensor_list[i].data_ptr(), POINTER(c_float)) for i in range(8)]
decompressed_tensor_ptr_list_c = (POINTER(c_float) * 8)(*decompressed_tensor_ptr_list)
decompressed_tensor_ptr_c = pointer(decompressed_tensor_ptr_list_c)

# launch decompression
pfz.pfzDecompress(compressed_ptr,
                  offset_list_ptr,
                  offset_list_size,
                  gpu_index,
                  decompressed_tensor_ptr_c)

# verification
print(decompressed_tensor_list[0][:100])
print(decompressed_tensor_list[1][:100])
