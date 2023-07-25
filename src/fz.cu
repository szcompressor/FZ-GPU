#include <algorithm>
#include <chrono>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <sys/stat.h>
#include <thrust/copy.h>

#include "../include/kernel/lorenzo_var.cuh"
#include "../include/utils/cuda_err.cuh"

#define UINT32_BIT_LEN 32
// #define VERIFICATION
// #define DEBUG

long GetFileSize(std::string fidataTypeLename)
{
    struct stat stat_buf;
    int rc = stat(fidataTypeLename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

template <typename T>
T *read_binary_to_new_array(const std::string &fname, size_t dtype_dataTypeLen)
{
    std::ifstream ifs(fname.c_str(), std::ios::binary | std::ios::in);
    if (not ifs.is_open())
    {
        std::cerr << "fail to open " << fname << std::endl;
        exit(1);
    }
    auto _a = new T[dtype_dataTypeLen]();
    ifs.read(reinterpret_cast<char *>(_a), std::streamsize(dtype_dataTypeLen * sizeof(T)));
    ifs.close();
    return _a;
}

template <typename T>
void write_array_to_binary(const std::string &fname, T *const _a, size_t const dtype_dataTypeLen)
{
    std::ofstream ofs(fname.c_str(), std::ios::binary | std::ios::out);
    if (not ofs.is_open())
        return;
    ofs.write(reinterpret_cast<const char *>(_a), std::streamsize(dtype_dataTypeLen * sizeof(T)));
    ofs.close();
}

__global__ void compressionFusedKernel(
    const uint32_t *__restrict__ in,
    uint32_t *out,
    uint32_t *deviceOffsetCounter,
    uint32_t *deviceBitFlagArr,
    uint32_t *deviceStartPosition,
    uint32_t *deviceCompressedSize)
{
    // 32 x 32 data chunk size with one padding for each row, overall 4096 bytes per chunk
    __shared__ uint32_t dataChunk[32][33];
    __shared__ uint16_t byteFlagArray[257];
    __shared__ uint32_t bitflagArr[8];
    __shared__ uint32_t startPosition;

    uint32_t byteFlag = 0;
    uint32_t v;

    v = in[threadIdx.x + threadIdx.y * 32 + blockIdx.x * 1024];
    __syncthreads();

#ifdef DEBUG
    dataChunk[threadIdx.y][threadIdx.x] = v;
    if (threadIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 1)
    {
        printf("original data:\n");
        for (int tmpIdx = 0; tmpIdx < 32; tmpIdx++)
        {
            printf("%u\t", dataChunk[0][tmpIdx]);
        }
        printf("\n");
    }
#endif

#pragma unroll 32
    for (int i = 0; i < 32; i++)
    {
        dataChunk[threadIdx.y][i] = __ballot_sync(0xFFFFFFFFU, v & (1U << i));
    }
    __syncthreads();

#ifdef DEBUG
    if (threadIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 1)
    {
        printf("shuffled data:\n");
        for (int tmpIdx = 0; tmpIdx < 32; tmpIdx++)
        {
            printf("%u\t", dataChunk[0][tmpIdx]);
        }
        printf("\n");
    }
#endif

    // generate byteFlagArray
    if (threadIdx.x < 8)
    {
#pragma unroll 4
        for (int i = 0; i < 4; i++)
        {
            byteFlag |= dataChunk[threadIdx.x * 4 + i][threadIdx.y];
        }
        byteFlagArray[threadIdx.y * 8 + threadIdx.x] = byteFlag > 0;
    }
    __syncthreads();

    // generate bitFlagArray
    uint32_t buffer;
    if (threadIdx.y < 8)
    {
        buffer = byteFlagArray[threadIdx.y * 32 + threadIdx.x];
        bitflagArr[threadIdx.y] = __ballot_sync(0xFFFFFFFFU, buffer);
    }
    __syncthreads();

#ifdef DEBUG
    if (threadIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 1)
    {
        printf("bit flag array: %u\n", bitflagArr[0]);
    }
#endif

    // write back bitFlagArray to global memory
    if (threadIdx.x < 8 && threadIdx.y == 0)
    {
        deviceBitFlagArr[blockIdx.x * 8 + threadIdx.x] = bitflagArr[threadIdx.x];
    }

    int blockSize = 256;
    int tid = threadIdx.x + threadIdx.y * 32;

    // prefix summation, up-sweep
    int prefixSumOffset = 1;
#pragma unroll 8
    for (int d = 256 >> 1; d > 0; d = d >> 1)
    {
        if (tid < d)
        {
            int ai = prefixSumOffset * (2 * tid + 1) - 1;
            int bi = prefixSumOffset * (2 * tid + 2) - 1;
            byteFlagArray[bi] += byteFlagArray[ai];
        }
        __syncthreads();
        prefixSumOffset *= 2;
    }

    // clear the last element
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        byteFlagArray[blockSize] = byteFlagArray[blockSize - 1];
        byteFlagArray[blockSize - 1] = 0;
    }
    __syncthreads();

    // prefix summation, down-sweep
#pragma unroll 8
    for (int d = 1; d < 256; d *= 2)
    {
        prefixSumOffset >>= 1;
        if (tid < d)
        {
            int ai = prefixSumOffset * (2 * tid + 1) - 1;
            int bi = prefixSumOffset * (2 * tid + 2) - 1;

            uint32_t t = byteFlagArray[ai];
            byteFlagArray[ai] = byteFlagArray[bi];
            byteFlagArray[bi] += t;
        }
        __syncthreads();
    }

#ifdef DEBUG
    if (threadIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 1)
    {
        printf("byte flag array:\n");
        for (int tmpIdx = 0; tmpIdx < 32; tmpIdx++)
        {
            printf("%u\t", byteFlagArray[tmpIdx]);
        }
        printf("\n");
    }
#endif

    // use atomicAdd to reserve a space for compressed data chunk
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        startPosition = atomicAdd(deviceOffsetCounter, byteFlagArray[blockSize] * 4);
        deviceStartPosition[blockIdx.x] = startPosition;
        deviceCompressedSize[blockIdx.x] = byteFlagArray[blockSize];
    }
    __syncthreads();

    // write back the compressed data based on the startPosition
    int flagIndex = floorf(tid / 4);
    if (byteFlagArray[flagIndex + 1] != byteFlagArray[flagIndex])
    {
        out[startPosition + byteFlagArray[flagIndex] * 4 + tid % 4] = dataChunk[threadIdx.x][threadIdx.y];
    }
}

__global__ void decompressionFusedKernel(
    uint32_t *deviceInput,
    uint32_t *deviceOutput,
    uint32_t *deviceBitFlagArr,
    uint32_t *deviceStartPosition)
{
    // allocate shared byte flag array
    __shared__ uint32_t dataChunk[32][33];
    __shared__ uint16_t byteFlagArray[257];
    __shared__ uint32_t startPosition;

    // there are 32 x 32 uint32_t in this data chunk
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int bid = blockIdx.x;

    // transfer bit flag array to byte flag array
    uint32_t bitFlag = 0;
    if (threadIdx.x < 8 && threadIdx.y == 0)
    {
        bitFlag = deviceBitFlagArr[bid * 8 + threadIdx.x];
#pragma unroll 32
        for (int tmpInd = 0; tmpInd < 32; tmpInd++)
        {
            byteFlagArray[threadIdx.x * 32 + tmpInd] = (bitFlag & (1U << tmpInd)) > 0;
        }
    }
    __syncthreads();

    int prefixSumOffset = 1;
    int blockSize = 256;

    // prefix summation, up-sweep
#pragma unroll 8
    for (int d = 256 >> 1; d > 0; d = d >> 1)
    {
        if (tid < d)
        {
            int ai = prefixSumOffset * (2 * tid + 1) - 1;
            int bi = prefixSumOffset * (2 * tid + 2) - 1;
            byteFlagArray[bi] += byteFlagArray[ai];
        }
        __syncthreads();
        prefixSumOffset *= 2;
    }

    // clear the last element
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        byteFlagArray[blockSize] = byteFlagArray[blockSize - 1];
        byteFlagArray[blockSize - 1] = 0;
    }
    __syncthreads();

    // prefix summation, down-sweep
#pragma unroll 8
    for (int d = 1; d < 256; d *= 2)
    {
        prefixSumOffset >>= 1;
        if (tid < d)
        {
            int ai = prefixSumOffset * (2 * tid + 1) - 1;
            int bi = prefixSumOffset * (2 * tid + 2) - 1;

            uint32_t t = byteFlagArray[ai];
            byteFlagArray[ai] = byteFlagArray[bi];
            byteFlagArray[bi] += t;
        }
        __syncthreads();
    }

#ifdef DEBUG
    if (threadIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 1)
    {
        printf("decompressed byte flag array:\n");
        for (int tmpIdx = 0; tmpIdx < 32; tmpIdx++)
        {
            printf("%u\t", byteFlagArray[tmpIdx]);
        }
        printf("\n");
    }
#endif

    // initialize the shared memory to all 0
    dataChunk[threadIdx.y][threadIdx.x] = 0;
    __syncthreads();

    // get the start position
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        startPosition = deviceStartPosition[bid];
    }
    __syncthreads();

    // write back shuffled data to shared mem
    int byteFlagInd = tid / 4;
    if (byteFlagArray[byteFlagInd + 1] != byteFlagArray[byteFlagInd])
    {
        dataChunk[threadIdx.x][threadIdx.y] = deviceInput[startPosition + byteFlagArray[byteFlagInd] * 4 + tid % 4];
    }
    __syncthreads();

    // store the corresponding uint32 to the register buffer
    uint32_t buffer = dataChunk[threadIdx.y][threadIdx.x];
    __syncthreads();

    // bitshuffle (reverse)
#pragma unroll 32
    for (int i = 0; i < 32; i++)
    {
        dataChunk[threadIdx.y][i] = __ballot_sync(0xFFFFFFFFU, buffer & (1U << i));
    }
    __syncthreads();

#ifdef DEBUG
    if (threadIdx.y == 0 && threadIdx.x == 0 && blockIdx.x == 1)
    {
        printf("decomopressed data:\n");
        for (int tmpIdx = 0; tmpIdx < 32; tmpIdx++)
        {
            printf("%u\t", dataChunk[0][tmpIdx]);
        }
        printf("\n");
    }
#endif

    // write back to global memory
    deviceOutput[tid + bid * blockDim.x * blockDim.y] = dataChunk[threadIdx.y][threadIdx.x];
}

// int inputSize, int x, int y, int z, double eb

void fzCompress(float *deviceInput, uint8_t *deviceCompressed, int *outputSizePtr, int inputSize, int x, int y, int z, float eb)
{
    // defination of some basic variables
    auto inputDimension = dim3(x, y, z);
    auto dataTypeLen = int(inputSize / sizeof(float));
    float timeElapsed;
    uint32_t offsetSum;

    // defination of device pointers
    uint16_t *deviceCompressedOutput;
    uint32_t *deviceBitFlagArr;
    uint32_t *deviceStartPosition;
    uint8_t *deviceCompressedStartPosition;
    deviceCompressedStartPosition = deviceCompressed + sizeof(int) * 5;

    bool *deviceSignNum;
    uint16_t *deviceQuantizationCode;
    uint32_t *deviceOffsetCounter;
    uint32_t *deviceCompressedSize;

    // defination of timers
    std::chrono::time_point<std::chrono::system_clock> compressionStart, compressionEnd;

    // to calculate some usefule constants
    int blockSize = 16;
    auto quantizationCodeByteLen = dataTypeLen * 2; // quantization code length in unit of bytes
    quantizationCodeByteLen = quantizationCodeByteLen % 4096 == 0 ? quantizationCodeByteLen : quantizationCodeByteLen - quantizationCodeByteLen % 4096 + 4096;
    auto paddingDataTypeLen = quantizationCodeByteLen / 2;
    int bitFlagArrSize = quantizationCodeByteLen % (blockSize * UINT32_BIT_LEN) == 0 ? quantizationCodeByteLen / (blockSize * UINT32_BIT_LEN) : int(quantizationCodeByteLen / (blockSize * UINT32_BIT_LEN)) + 1;

    dim3 block(32, 32);
    dim3 grid(floor(paddingDataTypeLen / 2048)); // divided by 2 is because the file is transformed from uint32 to uint16

    CHECK_CUDA(cudaMalloc((void **)&deviceQuantizationCode, sizeof(uint16_t) * paddingDataTypeLen));
    CHECK_CUDA(cudaMalloc((void **)&deviceSignNum, sizeof(bool) * paddingDataTypeLen));

    CHECK_CUDA(cudaMalloc((void **)&deviceOffsetCounter, sizeof(uint32_t)));
    CHECK_CUDA(cudaMalloc((void **)&deviceCompressedSize, sizeof(uint32_t) * floor(quantizationCodeByteLen / 4096)));

    // cuda copy some info to compressed data
    CHECK_CUDA(cudaMemcpy(deviceCompressed, &inputSize, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceCompressed + sizeof(int), &x, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceCompressed + sizeof(int) * 1, &y, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceCompressed + sizeof(int) * 2, &z, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(deviceCompressed + sizeof(int) * 3, &eb, sizeof(int), cudaMemcpyHostToDevice));

    // get the differents by the calculated offset
    int offsetCalculator = 0;
    deviceCompressedOutput = (uint16_t *)(deviceCompressedStartPosition + offsetCalculator);
    offsetCalculator += sizeof(uint16_t) * paddingDataTypeLen;
    deviceBitFlagArr = (uint32_t *)(deviceCompressedStartPosition + offsetCalculator);
    offsetCalculator += sizeof(uint32_t) * bitFlagArrSize;
    deviceStartPosition = (uint32_t *)(deviceCompressedStartPosition + offsetCalculator);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    compressionStart = std::chrono::system_clock::now();

    // pre-quantization
    cusz::experimental::launch_construct_LorenzoI_var<float, uint16_t, float>(deviceInput, deviceQuantizationCode, deviceSignNum, inputDimension, eb, timeElapsed, stream);

    // bitshuffle kernel
    compressionFusedKernel<<<grid, block>>>((uint32_t *)deviceQuantizationCode, (uint32_t *)deviceCompressedOutput, deviceOffsetCounter, deviceBitFlagArr, deviceStartPosition, deviceCompressedSize);

    cudaDeviceSynchronize();
    compressionEnd = std::chrono::system_clock::now();

#ifdef VERIFICATION

    uint16_t *hostQuantizationCode;
    hostQuantizationCode = (uint16_t *)malloc(sizeof(uint16_t) * dataTypeLen);
    CHECK_CUDA(cudaMemcpy(hostQuantizationCode, deviceQuantizationCode, sizeof(uint16_t) * dataTypeLen, cudaMemcpyDeviceToHost));

    // bitshuffle verification
    uint16_t *hostDecompressedQuantizationCode;
    hostDecompressedQuantizationCode = (uint16_t *)malloc(sizeof(uint16_t) * dataTypeLen);
    CHECK_CUDA(cudaMemcpy(hostDecompressedQuantizationCode, deviceDecompressedQuantizationCode, sizeof(uint16_t) * dataTypeLen, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();

    printf("begin bitshuffle verification\n");
    bool bitshuffleVerify = true;
    for (int tmpIdx = 0; tmpIdx < dataTypeLen; tmpIdx++)
    {
        if (hostQuantizationCode[tmpIdx] != hostDecompressedQuantizationCode[tmpIdx])
        {
            printf("data type len: %u\n", dataTypeLen);
            printf("verification failed at index: %d\noriginal quantization code: %u\ndecompressed quantization code: %u\n", tmpIdx, hostQuantizationCode[tmpIdx], hostDecompressedQuantizationCode[tmpIdx]);
            bitshuffleVerify = false;
            break;
        }
    }

    free(hostQuantizationCode);
    free(hostDecompressedQuantizationCode);

    // pre-quantization verification
    float *hostDecompressedOutput;
    hostDecompressedOutput = (float *)malloc(sizeof(float) * dataTypeLen);
    CHECK_CUDA(cudaMemcpy(hostDecompressedOutput, deviceDecompressedOutput, sizeof(float) * dataTypeLen, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();

    bool prequantizationVerify = true;
    if (bitshuffleVerify)
    {
        printf("begin pre-quantization verification\n");
        for (int tmpIdx = 0; tmpIdx < dataTypeLen; tmpIdx++)
        {
            if (std::abs(hostInput[tmpIdx] - hostDecompressedOutput[tmpIdx]) > float(eb * 1.01))
            {
                printf("verification failed at index: %d\noriginal data: %f\ndecompressed data: %f\n", tmpIdx, hostInput[tmpIdx], hostDecompressedOutput[tmpIdx]);
                printf("error is: %f, while error bound is: %f\n", std::abs(hostInput[tmpIdx] - hostDecompressedOutput[tmpIdx]), float(eb));
                prequantizationVerify = false;
                break;
            }
        }
    }

    free(hostDecompressedOutput);

    // print verification result
    if (bitshuffleVerify)
    {
        printf("bitshuffle verification succeed!\n");
        if (prequantizationVerify)
        {
            printf("pre-quantization verification succeed!\n");
        }
        else
        {
            printf("pre-quantization verification fail\n");
        }
    }
    else
    {
        printf("bitshuffle verification fail\n");
    }

#endif

    CHECK_CUDA(cudaMemcpy(&offsetSum, deviceOffsetCounter, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    printf("original size: %d\n", inputSize);
    printf("compressed size: %ld\n", sizeof(uint32_t) * bitFlagArrSize + offsetSum * sizeof(uint32_t) + sizeof(uint32_t) * int(quantizationCodeByteLen / 4096));
    printf("compression ratio: %f\n", float(inputSize) / float(sizeof(uint32_t) * bitFlagArrSize + offsetSum * sizeof(uint32_t) + sizeof(uint32_t) * floor(quantizationCodeByteLen / 4096)));
    *outputSizePtr = sizeof(uint32_t) * bitFlagArrSize + offsetSum * sizeof(uint32_t) + sizeof(uint32_t) * int(quantizationCodeByteLen / 4096) + 5;

    std::chrono::duration<double> compressionTime = compressionEnd - compressionStart;

    std::cout << "compression e2e time: " << compressionTime.count() << " s\n";
    std::cout << "compression e2e throughput: " << float(inputSize) / 1024 / 1024 / 1024 / compressionTime.count() << " GB/s\n";

    CHECK_CUDA(cudaFree(deviceQuantizationCode));
    CHECK_CUDA(cudaFree(deviceSignNum));

    CHECK_CUDA(cudaFree(deviceOffsetCounter));
    CHECK_CUDA(cudaFree(deviceCompressedSize));

    cudaStreamDestroy(stream);

    return;
}

void fzDecompress(uint8_t *deviceCompressed, float *deviceDecompressedOutput)
{
    // define the input info variables
    int inputSize = 0;
    int x = 0;
    int y = 0;
    int z = 0;
    float eb = 0;

    // copy the input information from the GPU global memory
    CHECK_CUDA(cudaMemcpy(&inputSize, deviceCompressed, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&x, deviceCompressed + 1, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&y, deviceCompressed + 2, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&z, deviceCompressed + 3, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&eb, deviceCompressed + 4, sizeof(float), cudaMemcpyDeviceToHost));

    uint8_t *deviceCompressedStartPosition;
    deviceCompressedStartPosition = deviceCompressed + sizeof(int) * 5;

    auto inputDimension = dim3(x, y, z);
    auto dataTypeLen = int(inputSize / sizeof(float));

    float timeElapsed;

    uint16_t *deviceDecompressedQuantizationCode;

    uint16_t *deviceCompressedOutput;
    uint32_t *deviceBitFlagArr;
    uint32_t *deviceStartPosition;

    bool *deviceSignNum;

    std::chrono::time_point<std::chrono::system_clock> decompressionStart, decompressionEnd;

    int blockSize = 16;
    auto quantizationCodeByteLen = dataTypeLen * 2; // quantization code length in unit of bytes
    quantizationCodeByteLen = quantizationCodeByteLen % 4096 == 0 ? quantizationCodeByteLen : quantizationCodeByteLen - quantizationCodeByteLen % 4096 + 4096;
    auto paddingDataTypeLen = quantizationCodeByteLen / 2;
    int bitFlagArrSize = quantizationCodeByteLen % (blockSize * UINT32_BIT_LEN) == 0 ? quantizationCodeByteLen / (blockSize * UINT32_BIT_LEN) : int(quantizationCodeByteLen / (blockSize * UINT32_BIT_LEN)) + 1;

    dim3 block(32, 32);
    dim3 grid(floor(paddingDataTypeLen / 2048)); // divided by 2 is because the file is transformed from uint32 to uint16

    CHECK_CUDA(cudaMalloc((void **)&deviceDecompressedQuantizationCode, sizeof(uint16_t) * paddingDataTypeLen));

    // get the differents by the calculated offset
    int offsetCalculator = 0;
    deviceCompressedOutput = (uint16_t *)(deviceCompressedStartPosition + offsetCalculator);
    offsetCalculator += sizeof(uint16_t) * paddingDataTypeLen;
    deviceBitFlagArr = (uint32_t *)(deviceCompressedStartPosition + offsetCalculator);
    offsetCalculator += sizeof(uint32_t) * bitFlagArrSize;
    deviceStartPosition = (uint32_t *)(deviceCompressedStartPosition + offsetCalculator);

    // CHECK_CUDA(cudaMemset(deviceDecompressedQuantizationCode, 0, sizeof(uint16_t) * paddingDataTypeLen));

    CHECK_CUDA(cudaMemset(deviceStartPosition, 0, sizeof(uint32_t) * floor(quantizationCodeByteLen / 4096)));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    decompressionStart = std::chrono::system_clock::now();

    // de-bitshuffle kernel
    decompressionFusedKernel<<<grid, block>>>((uint32_t *)deviceCompressedOutput, (uint32_t *)deviceDecompressedQuantizationCode, deviceBitFlagArr, deviceStartPosition);

    // de-pre-quantization
    cusz::experimental::launch_reconstruct_LorenzoI_var<float, uint16_t, float>(deviceSignNum, deviceDecompressedQuantizationCode, deviceDecompressedOutput, inputDimension, eb, timeElapsed, stream);

    cudaDeviceSynchronize();
    decompressionEnd = std::chrono::system_clock::now();

#ifdef VERIFICATION

    uint16_t *hostQuantizationCode;
    hostQuantizationCode = (uint16_t *)malloc(sizeof(uint16_t) * dataTypeLen);
    CHECK_CUDA(cudaMemcpy(hostQuantizationCode, deviceQuantizationCode, sizeof(uint16_t) * dataTypeLen, cudaMemcpyDeviceToHost));

    // bitshuffle verification
    uint16_t *hostDecompressedQuantizationCode;
    hostDecompressedQuantizationCode = (uint16_t *)malloc(sizeof(uint16_t) * dataTypeLen);
    CHECK_CUDA(cudaMemcpy(hostDecompressedQuantizationCode, deviceDecompressedQuantizationCode, sizeof(uint16_t) * dataTypeLen, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();

    printf("begin bitshuffle verification\n");
    bool bitshuffleVerify = true;
    for (int tmpIdx = 0; tmpIdx < dataTypeLen; tmpIdx++)
    {
        if (hostQuantizationCode[tmpIdx] != hostDecompressedQuantizationCode[tmpIdx])
        {
            printf("data type len: %u\n", dataTypeLen);
            printf("verification failed at index: %d\noriginal quantization code: %u\ndecompressed quantization code: %u\n", tmpIdx, hostQuantizationCode[tmpIdx], hostDecompressedQuantizationCode[tmpIdx]);
            bitshuffleVerify = false;
            break;
        }
    }

    free(hostQuantizationCode);
    free(hostDecompressedQuantizationCode);

    // pre-quantization verification
    float *hostDecompressedOutput;
    hostDecompressedOutput = (float *)malloc(sizeof(float) * dataTypeLen);
    CHECK_CUDA(cudaMemcpy(hostDecompressedOutput, deviceDecompressedOutput,
                          sizeof(float) * dataTypeLen, cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();

    bool prequantizationVerify = true;
    if (bitshuffleVerify)
    {
        printf("begin pre-quantization verification\n");
        for (int tmpIdx = 0; tmpIdx < dataTypeLen; tmpIdx++)
        {
            if (std::abs(hostInput[tmpIdx] - hostDecompressedOutput[tmpIdx]) > float(eb * 1.01))
            {
                printf("verification failed at index: %d\noriginal data: %f\ndecompressed data: %f\n", tmpIdx, hostInput[tmpIdx], hostDecompressedOutput[tmpIdx]);
                printf("error is: %f, while error bound is: %f\n", std::abs(hostInput[tmpIdx] - hostDecompressedOutput[tmpIdx]), float(eb));
                prequantizationVerify = false;
                break;
            }
        }
    }

    free(hostDecompressedOutput);

    // print verification result
    if (bitshuffleVerify)
    {
        printf("bitshuffle verification succeed!\n");
        if (prequantizationVerify)
        {
            printf("pre-quantization verification succeed!\n");
        }
        else
        {
            printf("pre-quantization verification fail\n");
        }
    }
    else
    {
        printf("bitshuffle verification fail\n");
    }

#endif

    std::chrono::duration<double> decompressionTime = decompressionEnd - decompressionStart;

    std::cout << "decompression e2e time: " << decompressionTime.count() << " s\n";
    std::cout << "decompression e2e throughput: " << float(inputSize) / 1024 / 1024 / 1024 / decompressionTime.count() << " GB/s\n";

    CHECK_CUDA(cudaFree(deviceStartPosition));
    CHECK_CUDA(cudaFree(deviceDecompressedQuantizationCode));

    cudaStreamDestroy(stream);

    // delete[] hostInput;

    return;
}

// List of Tensor(GPU),
// list of device index(CPU),
// list of input size(CPU),
// world size(int CPU),
// list of error bound(CPU),
// list of dimensions (CPU),
// output(GPU),
// Compressed Size Tensor(GPU/CPU)

extern "C"
{
    void pfzCompress(float **deviceInput,
                     int gpuIndex,
                     int *inputSizeArr,
                     int arrSize,
                     int worldSize,
                     float *errorBoundArr,
                     int **dimensionInfoArr,
                     uint8_t *deviceCompressed,
                     int **compressedSizeArr)
    {
        CHECK_CUDA(cudaSetDevice(gpuIndex));
        int chunkInputSizeArr[arrSize] = {0};
        for (int i = 0; i < arrSize; i++)
        {
            // get the data chunk size for each input tensor, we add paddings to the originla data so that it can be divided by the world size
            chunkInputSizeArr[i] = inputSizeArr[i] / worldSize == 0 ? inputSizeArr[i] / worldSize : inputSizeArr[i] / worldSize + 1;
        }

        int x, y, z;
        float eb;
        int outputSizeCounter = 0;
        int outputSize = 0;
        for (int i = 0; i < arrSize; i++)
        {
            for (int j = 0; j < worldSize; j++)
            {
                x = dimensionInfoArr[i][0];
                y = dimensionInfoArr[i][1];
                z = dimensionInfoArr[i][2];
                eb = errorBoundArr[i];
                int actualInputSize = j == (worldSize - 1) ? chunkInputSizeArr[i] - worldSize + inputSizeArr[i] % worldSize : chunkInputSizeArr[i];

                fzCompress(deviceInput[i] + chunkInputSizeArr[i] * j,
                           deviceCompressed + outputSizeCounter,
                           &outputSize,
                           actualInputSize,
                           x, y, z, eb);

                outputSizeCounter += outputSize;
                compressedSizeArr[i][j] = outputSize;
            }
        }

        return;
    }

    void pfzDecompress(uint8_t *deviceCompressed,
                       int *offsetArr,
                       int offsetArrSize,
                       int gpuIndex,
                       float **deviceDecompressedOutput)
    {
        CHECK_CUDA(cudaSetDevice(gpuIndex));
        for (int i = 0; i < offsetArrSize; i++)
        {
            fzDecompress(deviceCompressed + offsetArr[i],
                         deviceDecompressedOutput[i]);
        }

        return;
    }
}
