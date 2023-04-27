main: fz.cu
	nvcc src/claunch_cuda.cu -Iinclude --extended-lambda -c
	nvcc src/fz.cu claunch_cuda.o -o fz-gpu

clean:
	rm -rf fz-gpu claunch_cuda.o
