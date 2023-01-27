# FZ-GPU: A Fast and High-Ratio Lossy Compressor for Scientific Data on GPUs

This software is implemented and optimized based on the [cuSZ framework](https://github.com/szcompressor/cuSZ). Specifically, we modified cuSZ’s dual-quantization kernel in kernel/lorenzo_var.cuh and implement our bitshuffle and new lossless encode kernels in fz.cu. 

## Environment
- NVIDIA GPUs
- GCC (9.3.0)
- CUDA (11.4.120)

## Compile
Please use compile.sh to compile FZ-GPU and you will get the executable ```fz-gpu```.

## Download Data
Please use ```get_sample_data.sh``` to download the sample data or more datasets from [SDRBench](http://sdrbench.github.io/).

```
./get_sample_data.sh
```

## Test
Please use the below command to test ```fz-gpu``` on the example float32 data.
```
./fz-gpu [input data path] [dimension z] [dimension y] [dimension x] [error bound]
```

For example,
```
./fz-gpu cesm-CLDHGH-3600x1800 3600 1800 1 1e-3
./fz-gpu exafel-59200x388 39200 388 1 1e-3
./fz-gpu hurr-CLOUDf48-500x500x100 500 500 100 1e-3
```

Finally, you will observe the output including compression ratio.
```
original size: 25920000
compressed size: 3105556
compression ratio: 8.346332
```

To get the time of our compression, please add ```nsys``` before the execution command, like
```
nsys profile --stats=true ./fz-gpu cesm-CLDHGH-3600x1800 3600 1800 1 1e-3
nsys profile --stats=true ./fz-gpu exafel-59200x388 39200 388 1 1e-3
nsys profile --stats=true ./fz-gpu hurr-CLOUDf48-500x500x100 500 500 100 1e-3
```

You will observe the time of our each kernel, i.e., cusz::experimental::c_lorenzo_1d/2d/3d (optimized Lorenzo kernel), bitshuffleAndBitflag (bitshuffle kernel), encodeDebug (encode kernel), cub::DeviceScanInitKernel, cub::DeviceScanKernel (prefix-sum kernels). 
