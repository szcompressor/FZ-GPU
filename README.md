# FZ-GPU: A Fast and High-Ratio Lossy Compressor for Scientific Data on GPUs

This software is implemented and optimized based on the [cuSZ framework](https://github.com/szcompressor/cuSZ). Specifically, we modified cuSZ’s dual-quantization kernel in kernel/lorenzo_var.cuh and implement our bitshuffle and new lossless encode kernels in fz.cu. FZ-GPU performs compression and decompression together (we will provide options to do compression and decompression separately in the future).

## Environment
- NVIDIA GPUs
- GCC (9.3.0)
- CUDA (11.4.120)

## Compile
Please use the following command to compile FZ-GPU and you will get the executable ```fz-gpu```.
```
make -j
```

## Download Data
Please use ```get_sample_data.sh``` to download the sample data or more datasets from [SDRBench](http://sdrbench.github.io/).

```
./get_sample_data.sh
```

## Run FZ-GPU
Please use the below command to test ```fz-gpu``` on the example float32 data.
```
./fz-gpu [input data path] [dimension x] [dimension y] [dimension z] [error bound]
```

For example,
```
./fz-gpu cesm-CLDHGH-3600x1800 3600 1800 1 1e-3
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

You will observe the time of our each kernel, i.e., cusz::experimental::c_lorenzo_1d/2d/3d (optimized Lorenzo kernel), compressionFusedKernel (fused compression kernel), cusz::experimental::x_lorenzo_1d/2d/3d (Lorenzo decompression kernel), decompressionFusedKernel (fused decompression kernel). 

## Citing FZ-GPU
**HPDC '23: FZ-GPU** ([local copy](HPDC23-FZ-GPU.pdf), [via ACM](https://dl.acm.org/doi/10.1145/3588195.3592994), or [via arXiv](https://arxiv.org/abs/2304.12557))

```bibtex
@inproceedings{fz2023zhang,
      title = {FZ-GPU: A Fast and High-Ratio Lossy Compressor for Scientific
Computing Applications on GPUs},
     author = {Zhang, Boyuan and Tian, Jiannan and Di, Sheng and Yu, Xiaodong and Feng, Yunhe and Liang, Xin and Tao, Dingwen and Cappello, Franck},
       year = {2023},
       isbn = {979-8-4007-0155-9/23/06},
  publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
	url = {https://dl.acm.org/doi/10.1145/3588195.3592994},
        doi = {10.1145/3588195.3592994},
  booktitle = {Proceedings of the 32nd International Symposium on High-Performance Parallel and Distributed Computing},
   numpages = {14},
   keywords = {Lossy compression; scientific data; GPU; performance},
   location = {Orlando, FL, USA},
     series = {HPDC '23}
}

## Acknowledgements
This R&D is supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations – the Office of Science and the National Nuclear Security Administration, responsible for the planning and preparation of a capable exascale ecosystem. This repository is based upon work supported by the U.S. Department of Energy, Office of Science, under contract DE-AC02-06CH11357, and also supported by the National Science Foundation under Grants OAC-2003709/2303064, OAC-2104023/2247080, and OAC-2312673.
