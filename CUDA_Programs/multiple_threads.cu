#include <stdio.h>

// Kernel Function that runs on the GPU
__global__ void myKernel() {
    // Each thread prints its ID
    printf("Hello from GPU! Thread ID: %d\n", threadIdx.x);
}

int main() {
    // Launch the Kernel with 1 block and 4 threads
    myKernel<<<1, 4>>>();

    // Wait for GPU execution to finish
    cudaDeviceSynchronize();

    return 0;
}
