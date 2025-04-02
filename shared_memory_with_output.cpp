#include <hip/hip_runtime.h>
#include <stdio.h>

#define N 256

// Kernel function
__global__ void shared_memory_example(int *a, int *b, int n) {
    __shared__ int sharedData[N];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        sharedData[threadIdx.x] = a[idx];
        __syncthreads();
        b[idx] = sharedData[threadIdx.x] * 2;
    }
}

int main() {
    int h_a[N], h_b[N];  // Host arrays
    int *d_a, *d_b;  // Device pointers

    // Initialize input data on host
    for (int i = 0; i < N; i++)
        h_a[i] = i;  // Fill with values: {0, 1, 2, ..., 255}

    // Allocate memory on GPU
    hipMalloc(&d_a, N * sizeof(int));
    hipMalloc(&d_b, N * sizeof(int));

    // Copy input data from host to device
    hipMemcpy(d_a, h_a, N * sizeof(int), hipMemcpyHostToDevice);

    // Launch kernel
    shared_memory_example<<<1, N>>>(d_a, d_b, N);

    // Copy result back to host
    hipMemcpy(h_b, d_b, N * sizeof(int), hipMemcpyDeviceToHost);

    // Print output
    printf("Output:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_b[i]);
    }
    printf("\n");

    // Free GPU memory
    hipFree(d_a);
    hipFree(d_b);

    return 0;
}

