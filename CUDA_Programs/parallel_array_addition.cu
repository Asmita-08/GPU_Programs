#include <stdio.h>
#include <cuda_runtime.h>

#define N 8  // Number of elements in the array

// CUDA Kernel for parallel addition
__global__ void addArrays(int *a, int *b, int *c) {
    int idx = threadIdx.x;  // Get thread ID
    if (idx < N) {  // Ensure we don't access out of bounds
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int a[N], b[N], c[N];  // Arrays in CPU memory (host)
    int *d_a, *d_b, *d_c;  // Pointers for GPU memory (device)

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        a[i] = i;      // a = {0, 1, 2, 3, 4, 5, 6, 7}
        b[i] = i * 2;  // b = {0, 2, 4, 6, 8, 10, 12, 14}
    }

    // Allocate memory on GPU using cudaMalloc
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    // Copy arrays from CPU to GPU using cudaMemcpy
    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch Kernel with N threads
    addArrays<<<1, N>>>(d_a, d_b, d_c);

    // Copy result back from GPU to CPU
    cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("Array A: ");
    for (int i = 0; i < N; i++) printf("%d ", a[i]);
    printf("\n");

    printf("Array B: ");
    for (int i = 0; i < N; i++) printf("%d ", b[i]);
    printf("\n");

    printf("Sum (A+B): ");
    for (int i = 0; i < N; i++) printf("%d ", c[i]);
    printf("\n");

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
