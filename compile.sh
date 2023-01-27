nvcc src/kernel/claunch_cuda.cu -Iinclude --extended-lambda -c
nvcc main.cu claunch_cuda.o -o llapi_demo
