#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 100000
#define MAX_CONFIGS 100

__global__ void vector_add_hip(int *A, int *B, int *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int blocks_arr[MAX_CONFIGS];
int threads_arr[MAX_CONFIGS];
float timings_arr[MAX_CONFIGS];
float warmup_arr[MAX_CONFIGS];
int configCount = 0;

float minTime = 1e9;
int bestBlocks = 0;
int bestThreads = 0;
float firstConfigTime = 0.0;
float firstWarmupTime = 0.0;
bool firstConfigMeasured = false;
bool firstWarmupMeasured = false;

void hipCheckError(hipError_t err, const char *msg, int line) {
    if (err != hipSuccess) {
        fprintf(stderr, "HIP Error: %s at line %d: %s\n", msg, line, hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define HIP_CHECK(cmd) hipCheckError((cmd), #cmd, __LINE__)

void test_configuration(int blocks, int threadsPerBlock) {
    int *h_A = (int *)malloc(N * sizeof(int));
    int *h_B = (int *)malloc(N * sizeof(int));
    int *h_C = (int *)malloc(N * sizeof(int));

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host malloc failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N; i++) {
        h_A[i] = 1;
        h_B[i] = 2;
    }

    int *d_A, *d_B, *d_C;
    size_t size = N * sizeof(int);
    HIP_CHECK(hipMalloc(&d_A, size));
    HIP_CHECK(hipMalloc(&d_B, size));
    HIP_CHECK(hipMalloc(&d_C, size));

    HIP_CHECK(hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice));

    // --- Warm-up timing ---
    hipEvent_t warmup_start, warmup_stop;
    HIP_CHECK(hipEventCreate(&warmup_start));
    HIP_CHECK(hipEventCreate(&warmup_stop));

    HIP_CHECK(hipEventRecord(warmup_start));
    hipLaunchKernelGGL(vector_add_hip, dim3(1), dim3(32), 0, 0, d_A, d_B, d_C, N);
    HIP_CHECK(hipEventRecord(warmup_stop));
    HIP_CHECK(hipEventSynchronize(warmup_stop));

    float warmup_ms = 0;
    HIP_CHECK(hipEventElapsedTime(&warmup_ms, warmup_start, warmup_stop));

    if (!firstWarmupMeasured) {
        firstWarmupTime = warmup_ms;
        firstWarmupMeasured = true;
        printf("\nFirst Warm-up Time: %.4f ms\n", firstWarmupTime);
    }

    // --- Actual execution timing ---
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    hipLaunchKernelGGL(vector_add_hip, dim3(blocks), dim3(threadsPerBlock), 0, 0, d_A, d_B, d_C, N);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

    if (!firstConfigMeasured) {
        firstConfigTime = ms;
        firstConfigMeasured = true;
        printf("\nFirst Configuration Execution Time: %.4f ms\n\n", firstConfigTime);
    }

    HIP_CHECK(hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost));

    bool correct = true;
    for (int i = 0; i < 10; ++i) {
        if (h_C[i] != 3) {
            correct = false;
            break;
        }
    }

    if (configCount < MAX_CONFIGS) {
        blocks_arr[configCount] = blocks;
        threads_arr[configCount] = threadsPerBlock;
        timings_arr[configCount] = ms;
        warmup_arr[configCount] = warmup_ms;

        if (correct && ms < minTime) {
            minTime = ms;
            bestBlocks = blocks;
            bestThreads = threadsPerBlock;
        }

        configCount++;
    }

    printf("Blocks: %4d | Threads/Block: %4d | Warm-up: %8.4f ms | Execution Time: %8.4f ms | Correct: %s\n",
           blocks, threadsPerBlock, warmup_ms, ms, correct ? "Yes" : "No");

    HIP_CHECK(hipFree(d_A)); HIP_CHECK(hipFree(d_B)); HIP_CHECK(hipFree(d_C));
    free(h_A); free(h_B); free(h_C);
}

int main() {
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    printf("Detected %d HIP device(s).\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, i));
        printf("Device %d: %s\n", i, prop.name);
        printf("  Total Global Mem: %lu MB\n", prop.totalGlobalMem / (1024 * 1024));
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Multiprocessors (SMs): %d\n\n", prop.multiProcessorCount);
    }

    for (int threads = 32; threads <= 1024; threads *= 2) {
        for (int blocks = 1; blocks <= N / threads; blocks *= 2) {
            test_configuration(blocks, threads);
        }
    }

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

    printf("warmup_arr = [");
    for (int i = 0; i < configCount; i++) {
        printf("%.4f", warmup_arr[i]);
        if (i < configCount - 1) printf(", ");
    }
    printf("]\n");

    printf("\n Optimal Configuration:\n");
    printf("   Blocks: %d\n", bestBlocks);
    printf("   Threads/Block: %d\n", bestThreads);
    printf("   Best Execution Time: %.4f ms\n", minTime);

    return 0;
}


