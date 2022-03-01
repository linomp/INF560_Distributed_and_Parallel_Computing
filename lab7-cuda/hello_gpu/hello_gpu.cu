#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda.h>

// Size of input data & result buffers
#define BUF 30

/* Function computing the final string to print */
__global__ void ComputeString(char *res, char *a, char *b, char *c, int length)
{
    int i;

    for (i = 0; i < length; i++)
    {
        res[i] = a[i] + b[i] + c[i];
    }
}

int main()
{
    // Arrays declaration in host memory
    char a[BUF] = {40, 70, 70, 70, 80, 0, 50, 80, 80, 70, 70, 0, 40, 80, 79, 70, 0, 40, 50, 50, 0, 70, 80, 0, 30, 50, 30, 30, 0, 0};
    char b[BUF] = {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                   10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0};
    char c[BUF] = {22, 21, 28, 28, 21, 22, 27, 21, 24, 28, 20, 22, 20, 24, 22,
                   29, 22, 21, 20, 25, 22, 25, 20, 22, 27, 25, 28, 25, 0, 0};

    // Results array allocation in host memory
    char *res;
    res = (char *)malloc(BUF * sizeof(char));

    // Allocate arrays in device memory
    size_t size = BUF * sizeof(char);

    char *d_a;
    cudaMalloc(&d_a, size);
    char *d_b;
    cudaMalloc(&d_b, size);
    char *d_c;
    cudaMalloc(&d_c, size);
    char *d_res;
    cudaMalloc(&d_res, size);

    // Copy array data from host memory to device memory (res is not necessary)
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);

    // TODO: threads?
    int N = BUF;
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    ComputeString<<<blocksPerGrid, threadsPerBlock>>>(d_res, d_a, d_b, d_c, BUF);

    // Copy result from device memory to host memory
    cudaMemcpy(res, d_res, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_res);

    // Report result & free host memory
    printf("%s\n", res);
    free(res);

    return 0;
}
