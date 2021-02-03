default: cuda-utilities
cuda-utilities: cuda-utilities.cu
	nvcc -O3 -Xcompiler "-Wall" -g -G cuda-utilities.cu -o cuda-utilities
clean:
	rm -f cuda-utilities