#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void hello_world() {
    printf("Hello, World from GPU!\n");
}

int main() {
    hello_world<<<1, 1>>>();  // Kernel launch
    hipDeviceSynchronize();   // Ensure kernel execution completes
    return 0;
}
