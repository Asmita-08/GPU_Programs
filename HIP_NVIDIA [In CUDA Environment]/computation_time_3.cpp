#include <hip/hip_runtime.h>
#include <stdio.h>
#include <math.h>

#define N 4096  // Large matrix size

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

// Function to test different configurations
void test_configuration(int threads_per_block_x, int threads_per_block_y) {
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
    int num_blocks_x = ceil((float)N / threads_per_block_x);
    int num_blocks_y = ceil((float)N / threads_per_block_y);
    dim3 threadsPerBlock(threads_per_block_x, threads_per_block_y);
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
    printf("Blocks: (%d, %d), Threads: (%d, %d) -> Execution Time: %f ms\n", 
           num_blocks_x, num_blocks_y, threads_per_block_x, threads_per_block_y, milliseconds);

    // Free memory
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

int main() {
    // Testing different block and thread configurations
    test_configuration(16, 16);
    test_configuration(32, 32);
    test_configuration(64, 64);
    test_configuration(128, 128);
    test_configuration(256, 256);
    
    return 0;
}

