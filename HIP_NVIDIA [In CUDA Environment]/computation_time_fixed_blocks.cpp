#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 4096  // Matrix Size (4096 x 4096)
#define BLOCKS 64  // Fixed number of blocks

// HIP Kernel for Matrix Multiplication
__global__ void matrix_mult_hip(int *A, int *B, int *C, int n) {
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Function to test different thread configurations
void test_configuration(int threads_per_block) {
    int *d_A, *d_B, *d_C;
    int *h_A, *h_B, *h_C;
    int size = N * N * sizeof(int);

    // Allocate memory
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1;
        h_B[i] = 1;
    }

    // Copy data to GPU
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    // Compute grid size dynamically
    int num_blocks_x = BLOCKS;
    int num_blocks_y = BLOCKS;
    dim3 threadsPerBlock(threads_per_block, threads_per_block);
    dim3 blocksPerGrid(num_blocks_x, num_blocks_y);

    // Measure GPU execution time
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    hipEventRecord(start);
    hipLaunchKernelGGL(matrix_mult_hip, blocksPerGrid, threadsPerBlock, 0, 0, d_A, d_B, d_C, N);
    hipEventRecord(stop);
    
    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);
    
    hipEventSynchronize(stop);
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);

    // Print Execution Time
    printf("Threads: (%d, %d) -> Execution Time: %f ms\n", 
           threads_per_block, threads_per_block, milliseconds);

    // Free memory
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

int main() {
    // Testing different thread configurations with fixed blocks
    test_configuration(16);
    test_configuration(32);
    test_configuration(64);
    test_configuration(128);
    test_configuration(256);
    test_configuration(512);
    test_configuration(1024);

    return 0;
}

