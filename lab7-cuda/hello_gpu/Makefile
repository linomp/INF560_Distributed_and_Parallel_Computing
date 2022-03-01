all:hello_gpu

hello_gpu: hello_gpu.cu
	nvcc -I. -o hello_gpu hello_gpu.cu	

clean:
	rm -f hello_gpu hello_gpu.exp hello_gpu.lib
