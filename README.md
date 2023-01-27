# FZ-GPU: A Fast and High-Ratio Lossy Compressor for Scientific Data on GPUs

This software is implemented and optimized based on the [cuSZ framework](https://github.com/szcompressor/cuSZ). Specifically, we modified cuSZ’s dual-quantization kernel in kernel/lorenzo_var.cuh and implement our bitshuffle and new lossless encode kernels in fz.cu. 

## Environment
- NVIDIA GPUs
- GCC (9.3.0)
- CUDA (11.4.120)

## Compile
Please use compile.sh to compile FZ-GPU and you will get the executable ```fz-gpu```.

## Download Data
Please download the datasets from [SDRBench](http://sdrbench.github.io/).

## Test
Please use the below command to test ```fz-gpu``` on float32 input data. 
```
./fz-gpu [input data path] [dimension z] [dimension y] [dimension x] [error bound]
```

For example,
```
./fz-gpu /data/HACC/xx.f32 280953867 1 1 1e-3
./fz-gpu /data/CESM/TSMX_1_1800_3600.f32 3600 1800 1 1e-3
./fz-gpu /data/Hurricane/Wf48.bin.f32 500 500 100 1e-3
```
