# FZ-GPU: A Fast and High-Ratio Lossy Compressor for Scientific Data on GPUs

This software is implemented and optimized based on the [cuSZ](https://github.com/szcompressor/cuSZ) framework. Specifically, FZ-GPU modifies cuSZ's dual-quantization kernel and implements a fused kernel containing the bitshuffle operation and a new lossless encoder. Currently, FZ-GPU performs compression and decompression together, but we plan to provide options for performing compression and decompression separately in the future.

(C) 2023 by Indiana University and Argonne National Laboratory.

- Developers: Boyuan Zhang, Jiannan Tian
- Contributors (alphabetic): Dingwen Tao, Franck Cappello, Sheng Di, Xiaodong Yu

## Recommended Environment
- Linux OS with NVIDIA GPUs
- GCC (>= 7.3.0)
- CUDA (>= 11.0)

## Compile
Please use the following command to compile FZ-GPU. You will get the executable ```fz-gpu```.
```
make -j
```

## Download Data
Please use ```get_sample_data.sh``` to download the sample data. More datasets can be downloaded from [SDRBench](http://sdrbench.github.io/).

```
./get_sample_data.sh
```

## Run FZ-GPU
Please use the below command to run ```fz-gpu``` on a float32 data.
```
./fz-gpu [input data path] [dimension x] [dimension y] [dimension z] [error bound]
```

For example,
```
./fz-gpu cesm-CLDHGH-3600x1800 3600 1800 1 1e-3
./fz-gpu hurr-CLOUDf48-500x500x100 500 500 100 1e-3
```

Finally, you can observe the output including compression ratio, compression/decompression end-to-end times, and compression/decompression end-to-end throughputs.
```
compressed size: 8975636
compression ratio: 21.113260
compression time: 0.000985528 s
compression e2e throughput: 179.082 GB/s
decompression time: 0.00114546 s
decompression e2d throughput: 154.079 GB/s
```

To obtain more accurate timing for the compression kernel, please use ```nsys``` before the execution command, like
```
nsys profile --stats=true ./fz-gpu cesm-CLDHGH-3600x1800 3600 1800 1 1e-3
nsys profile --stats=true ./fz-gpu exafel-59200x388 39200 388 1 1e-3
nsys profile --stats=true ./fz-gpu hurr-CLOUDf48-500x500x100 500 500 100 1e-3
```

You will observe the time for each kernel, i.e., cusz::experimental::c_lorenzo_1d/2d/3d (optimized Lorenzo kernel), compressionFusedKernel (fused compression kernel), cusz::experimental::x_lorenzo_1d/2d/3d (Lorenzo reconstruction kernel), and decompressionFusedKernel (fused decompression kernel).

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
```

## Acknowledgements
This R&D is supported by the Exascale Computing Project (ECP), Project Number: 17-SC-20-SC, a collaborative effort of two DOE organizations – the Office of Science and the National Nuclear Security Administration, responsible for the planning and preparation of a capable exascale ecosystem. This repository is based upon work supported by the U.S. Department of Energy, Office of Science, under contract DE-AC02-06CH11357, and also supported by the National Science Foundation under Grants OAC-2003709/2303064, OAC-2104023/2247080, and OAC-2312673.
