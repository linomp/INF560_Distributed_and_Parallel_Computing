#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda.h>

// Workaround to test locally on my machine (windows)
#ifdef _WIN32
long lrand48()
{
    return rand();
}

void srand48(long seedval)
{
    srand(seedval);
}
#endif

// Device code
__global__ void VecAdd(int *A, int *B, int *C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Host code
int main(int argc, char **argv)
{
    // Take two input parameters (through command line):
    // N for the number of elements in vector and S for the seed used to fill the two input vectors
    int N;
    int S;

    // Check the input arguments
    if (argc < 3)
    {
        printf("Usage: %s S N\n", argv[0]);
        printf("\tS: seed for pseudo-random generator\n");
        printf("\tN: size of the array\n");
        exit(1);
    }

    S = atoi(argv[1]);
    N = atoi(argv[2]);
    srand48(S);

    // Allocate input vectors h_A and h_B in host memory
    size_t size = N * sizeof(int);

    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);

    // Allocate result vector in host memory
    int *h_res = (int *)malloc(size);

    // Initialize input vectors with random integer values (with the seed S)
    for (int i = 0; i < N; i++)
    {
        h_A[i] = lrand48() % N;
        h_B[i] = lrand48() % N;
    }

    // Allocate vectors in device memory
    int *d_A;
    cudaMalloc(&d_A, size);
    int *d_B;
    cudaMalloc(&d_B, size);
    int *d_res;
    cudaMalloc(&d_res, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_res, N);

    // Copy result from device memory to host memory
    cudaMemcpy(h_res, d_res, size, cudaMemcpyDeviceToHost);

    // Check that the resulting vector C is the sum of A and B
    float error = 0;
    float tol = 1e-6;
    int temp = 0;
    for (int j = 0; j < N; j++)
    {
        temp = h_A[j] + h_B[j];
        error += abs(h_res[j] - temp);
#ifdef DEBUG
        printf("temp = %d + %d = %d | h_res[j] = %d\n", h_A[j], h_B[j], temp, h_res[j]);
#endif
    }
    printf("Accumulated error: %f\n", error);
    printf("Sum is correct: %s\n", abs(error) <= tol ? "True" : "False");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_res);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_res);
}
