all:add_vectors

add_vectors: add_vectors.cu
	nvcc -I. -o add_vectors add_vectors.cu	

clean:
	rm -f add_vectors add_vectors.exp add_vectors.lib
