#include <hip/hip_runtime.h>
#include <stdio.h>

#define N 1024  // Matrix size (N x N)

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

// Function to test different configurations on a selected device
void test_configuration(int deviceId, int numBlocks, int numThreads) {
    hipSetDevice(deviceId);

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
    hipMemcpy(d_B, d_B, size, hipMemcpyHostToDevice);

    // Define kernel launch configuration
    dim3 threadsPerBlock(numThreads, numThreads);
    dim3 blocksPerGrid(numBlocks, numBlocks);

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
    printf("Device %d -> Blocks: %d, Threads: %d -> Execution Time: %f ms\n", deviceId, numBlocks, numThreads, milliseconds);

    // Free memory
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

int main() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    printf("Total number of HIP-compatible GPUs: %d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        printf("\nDevice ID: %d\n", i);
        printf("  Name: %s\n", prop.name);
        printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Threads Dim: [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Size: [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("\n Number of Streaming Multiprocessors (SMs) [in NVIDIA GPU] or Compute Units (CUs) [in AMD GPU]: %d ", prop.multiProcessorCount);
	printf("\n Warp Size [in NVIDIA GPU] or Wavefront Size [in AMD GPU]: %d\n", prop.warpSize); 

        // Run tests on this GPU
	printf("\n");
	test_configuration(i, 1, 1000);
	test_configuration(i, 1000, 1);
        test_configuration(i, 1, 32);
        test_configuration(i, 32, 32);
        test_configuration(i, 16, 64);
        test_configuration(i, 64, 16);
	test_configuration(i, 1, 64);
	test_configuration(i, 1, 16);
    }

    return 0;
}


