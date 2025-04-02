#include <cuda_runtime.h>
#include <stdio.h>

__global__ void shared_memory_example(int *a, int *b, int n) {
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
    int *h_a, *h_b, *d_a, *d_b;

    size_t size = n * sizeof(int);
    
    // Allocate memory
    h_a = (int*)malloc(size);
    h_b = (int*)malloc(size);
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);

    // Initialize input
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
    }

    // Copy to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // Launch kernel
    shared_memory_example<<<1, n>>>(d_a, d_b, n);

    // Copy back result
    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Shared Memory Example Output:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_b[i]);
    }
    printf("\n");

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);

    return 0;
}
