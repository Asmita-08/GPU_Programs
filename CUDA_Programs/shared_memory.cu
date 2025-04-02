#include <hip/hip_runtime.h>
#include <stdio.h>

__global__ void shared_memory(int *a, int *b, int n) {
    __shared__ int sharedData[256];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        sharedData[threadIdx.x] = a[idx];
        __syncthreads();
        b[idx] = sharedData[threadIdx.x] * 2;
    }
}

int main() {
    int n = 256;
    int *d_a, *d_b;
    hipMalloc(&d_a, n * sizeof(int));
    hipMalloc(&d_b, n * sizeof(int));

    shared_memory<<<1, n>>>(d_a, d_b, n);
    
    hipFree(d_a); hipFree(d_b);
    return 0;
}
