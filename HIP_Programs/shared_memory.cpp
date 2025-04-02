#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void shared_memory_example(int *a, int *b, int n) {
    __shared__ int sharedData[256];  // Shared memory array for one block
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Global thread index

    if (idx < n) {
        sharedData[threadIdx.x] = a[idx];  // Load global data into shared memory
        __syncthreads();  // Synchronize all threads in the block

        b[idx] = sharedData[threadIdx.x] * 2;  // Perform computation and store in global memory
    }
}


int main() {
    int n = 256;
    int *d_a, *d_b;

      hipMalloc(&d_a, n * sizeof(int));  // Allocate GPU memory for input array
    hipMalloc(&d_b, n * sizeof(int));  // Allocate GPU memory for output array

    shared_memory_example<<<1, n>>>(d_a, d_b, n);  // Launch kernel with 1 block, 256 threads

    hipFree(d_a); hipFree(d_b);  // Free GPU memory
    return 0;
}

