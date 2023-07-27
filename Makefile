main: src/fz.cu
	nvcc src/claunch_cuda.cu -Iinclude --extended-lambda -c
	nvcc -Xcompiler -fPIC -shared -g -G -o fz-gpu.so src/fz.cu

clean:
	rm -rf fz-gpu.so claunch_cuda.o
