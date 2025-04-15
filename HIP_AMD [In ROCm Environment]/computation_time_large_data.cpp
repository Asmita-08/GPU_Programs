#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 100000
#define MAX_CONFIGS 100  // Maximum number of configurations

// HIP kernel for vector addition
__global__ void vector_add_hip(int *A, int *B, int *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// Global arrays to store results
int blocks_arr[MAX_CONFIGS];
int threads_arr[MAX_CONFIGS];
float timings_arr[MAX_CONFIGS];
int configCount = 0;

// Track optimal configuration
float minTime = 1e9;
int bestBlocks = 0;
int bestThreads = 0;

// Function to test a specific block/thread configuration
void test_configuration(int blocks, int threadsPerBlock) {
    int *h_A = (int *)malloc(N * sizeof(int));
    int *h_B = (int *)malloc(N * sizeof(int));
    int *h_C = (int *)malloc(N * sizeof(int));

    int *d_A, *d_B, *d_C;
    size_t size = N * sizeof(int);

    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = 1;
        h_B[i] = 2;
     }

    // Allocate device memory
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);

       // Copy to device
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    // Time measurement
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    hipLaunchKernelGGL(vector_add_hip, dim3(blocks), dim3(threadsPerBlock), 0, 0, d_A, d_B, d_C, N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0;
    hipEventElapsedTime(&ms, start, stop);

    // Copy back to host
    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);

    // Check correctness
bool correct = true;
    for (int i = 0; i < 10; ++i) {
        if (h_C[i] != 3) {
            correct = false;
            break;
        }
    }

    // Store results
    if (configCount < MAX_CONFIGS) {
        blocks_arr[configCount] = blocks;
        threads_arr[configCount] = threadsPerBlock;
        timings_arr[configCount] = ms;

        // Track best
        if (correct && ms < minTime) {
            minTime = ms;
            bestBlocks = blocks;
            bestThreads = threadsPerBlock;
        }

        configCount++;
}

    printf("Blocks: %4d | Threads/Block: %4d | Time: %8.4f ms | Correct: %s\n",
           blocks, threadsPerBlock, ms, correct ? "Yes" : "No");

    // Free memory
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    free(h_A); free(h_B); free(h_C);
}

void warmup_gpu() {
    int *d_A;
    hipMalloc(&d_A, sizeof(int) * N);
    hipMemset(d_A, 0, sizeof(int) * N);
    hipFree(d_A);
}


int main() {

    // Warm up the GPU to avoid first-time overhead
    warmup_gpu();
 // GPU Info
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    printf("Detected %d HIP device(s).\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Total Global Mem: %lu MB\n", prop.totalGlobalMem / (1024 * 1024));
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Multiprocessors (SMs): %d\n\n", prop.multiProcessorCount);
    }

    // Run various configurations
    for (int threads = 32; threads <= 1024; threads *= 2) {
        for (int blocks = 1; blocks <= threads / 2; blocks *= 2) {
            test_configuration(blocks, threads);
        }
 }

    // Print arrays for plotting
    printf("\nblocks_arr = [");
    for (int i = 0; i < configCount; i++) {
        printf("%d", blocks_arr[i]);
        if (i < configCount - 1) printf(", ");
    }
    printf("]\n");

    printf("threads_arr = [");
    for (int i = 0; i < configCount; i++) {
        printf("%d", threads_arr[i]);
        if (i < configCount - 1) printf(", ");
    }
    printf("]\n");

    printf("timings_arr = [");
    for (int i = 0; i < configCount; i++) {
        printf("%.4f", timings_arr[i]);
        if (i < configCount - 1) printf(", ");
    }
 printf("]\n");

    // Show optimal
    printf("\n Optimal Configuration:\n");
    printf("   Blocks: %d\n", bestBlocks);
    printf("   Threads/Block: %d\n", bestThreads);
    printf("   Best Execution Time: %.4f ms\n", minTime);

    return 0;
}


