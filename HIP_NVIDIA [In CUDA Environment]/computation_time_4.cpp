#include <hip/hip_runtime.h>
#include <stdio.h>

#define N 4096  // Matrix size (4096 x 4096), Maximum N = 23170
#define TILE_SIZE 32  // Optimal thread block size (16x16 or 32x32), Maximum TILE_SIZE = 64

__global__ void matrix_mult_optimized(int *A, int *B, int *C, int n) {
    __shared__ int tileA[TILE_SIZE][TILE_SIZE];
    __shared__ int tileB[TILE_SIZE][TILE_SIZE];

    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    int sum = 0;
    for (int k = 0; k < n / TILE_SIZE; k++) {
        tileA[threadIdx.y][threadIdx.x] = A[row * n + (k * TILE_SIZE + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = B[(k * TILE_SIZE + threadIdx.y) * n + col];

        __syncthreads(); // Ensure tiles are loaded before computation

        for (int j = 0; j < TILE_SIZE; j++)
            sum += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];

        __syncthreads();
    }

    C[row * n + col] = sum;
}

void run_experiment(int blockSize) {
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

    // Copy to GPU
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid(N / blockSize, N / blockSize);

    // Measure GPU time
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    hipEventRecord(start);
    hipLaunchKernelGGL(matrix_mult_optimized, blocksPerGrid, threadsPerBlock, 0, 0, d_A, d_B, d_C, N);
    hipEventRecord(stop);
    
    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);
    
    hipEventSynchronize(stop);
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);

    // Print execution time
    printf("BlockSize: (%d, %d) -> Execution Time: %f ms\n", 
           blockSize, blockSize, milliseconds);

    // Free memory
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

int main() {
    // Test with different block sizes
    run_experiment(16);  // 16x16 threads per block
    run_experiment(32);  // 32x32 threads per block
    run_experiment(64);  // 64x64 threads per block (too large)
    run_experiment(128);
    run_experiment(256);
    run_experiment(512);
    run_experiment(1024);
    run_experiment(2048);
    run_experiment(4096);
    run_experiment(8192);
    run_experiment(16384);
    run_experiment(32768);

    return 0;
}

