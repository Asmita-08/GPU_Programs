#include<cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel function for parallel sum
__global__ void parallel_sum(int *input, int *output, int n) {
    __shared__ int temp[256];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    temp[threadIdx.x] = (idx < n) ? input[idx] : 0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            temp[threadIdx.x] += temp[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) output[blockIdx.x] = temp[0];
}

// Main function
int main() {
    int n = 256;  // Array size
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    int *h_input, *h_output;
    int *d_input, *d_output;

    h_input = (int*)malloc(n * sizeof(int));
    h_output = (int*)malloc(gridSize * sizeof(int));

    for (int i = 0; i < n; i++) {
        h_input[i] = 1;  // Initialize array with 1s
    }

    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, gridSize * sizeof(int));

    cudaMemcpy(d_input, h_input, n * sizeof(int), cudaMemcpyHostToDevice);

    parallel_sum<<<gridSize, blockSize>>>(d_input, d_output, n);

    cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    int total_sum = 0;
    for (int i = 0; i < gridSize; i++) {
        total_sum += h_output[i];
    }

    printf("Total sum: %d\n", total_sum);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}


