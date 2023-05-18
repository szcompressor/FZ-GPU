import numpy as np
import torch
import ctypes
from ctypes import *
from random import random

# create example tensors on GPU
input_tensor_gpu = torch.tensor([random() for i in range(1024 * 1024)], dtype=torch.float32).cuda()
output_tensor_gpu = torch.tensor([0 for i in range(1024 * 1024)], dtype=torch.float32).cuda()

# compression and decompression round trip
def pfz_round_trip():
    dll = ctypes.CDLL('./fz-gpu.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.fzRoundTrip
    func.argtypes = [POINTER(c_float), POINTER(c_float), c_int, c_int, c_int, c_int, c_double]
    return func

def run_pfz(input, output, x, y, z, error_bound):
    # get input GPU pointer
    input_gpu_ptr = input.data_ptr()
    input_gpu_ptr = cast(input_gpu_ptr, ctypes.POINTER(c_float))

    # get output GPU pointer
    output_gpu_ptr = output.data_ptr()
    output_gpu_ptr = cast(output_gpu_ptr, ctypes.POINTER(c_float))

    __pfz = pfz_round_trip()
    __pfz(input_gpu_ptr, output_gpu_ptr, c_int(x * y * z * 4), c_int(x), c_int(y), c_int(z), c_double(error_bound))

if __name__ == '__main__':
    run_pfz(input_tensor_gpu, output_tensor_gpu, 1024, 1024, 1, 1e-3)
