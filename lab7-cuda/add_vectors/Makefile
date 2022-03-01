all:add_vectors add_vectors_timed

add_vectors: add_vectors.cu
	nvcc -I. -o add_vectors add_vectors.cu	

add_vectors_timed: add_vectors_timed.cu
	nvcc -I. -o add_vectors_timed add_vectors_timed.cu	

clean:
	rm -rf add_vectors add_vectors.exp add_vectors.lib add_vectors_timed add_vectors_timed.exp add_vectors_timed.lib
