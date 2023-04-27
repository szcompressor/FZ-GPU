main: fz.cu
	nvcc src/kernel/claunch_cuda.cu -Iinclude --extended-lambda -c
	nvcc fz.cu claunch_cuda.o -o fz-gpu

clean:
	rm -rf fz-gpu claunch_cuda.o