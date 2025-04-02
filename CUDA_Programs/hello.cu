#include <cuda_runtime.h>
#include <stdio.h>

__global__ void hello_world() {
    printf("Hello, World from GPU!\n");
}

int main() {
    hello_world<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
