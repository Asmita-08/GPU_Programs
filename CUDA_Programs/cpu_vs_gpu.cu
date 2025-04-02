#include <stdio.h>
#include <cuda.h>
#include <chrono> // For timing

__global__
void deviceKernel(int *a, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride)
    {
        a[i] = 1;
    }
}

void hostFunction(int *a, int N)
{
    for (int i = 0; i < N; ++i)
    {
        a[i] = 1;
    }
}

int main()
{
    int N = 2 << 24;  // 33,554,432 elements
    size_t size = N * sizeof(int);
    int *a;

    cudaMallocManaged(&a, size);

    // Timing the CPU execution
    auto startCPU = std::chrono::high_resolution_clock::now();
    hostFunction(a, N);
    auto endCPU = std::chrono::high_resolution_clock::now();
    printf("CPU Execution Time: %f ms\n", std::chrono::duration<float, std::milli>(endCPU - startCPU).count());

    // Reset array before GPU execution
    for (int i = 0; i < N; ++i) a[i] = 0;

    // Timing the GPU execution
    auto startGPU = std::chrono::high_resolution_clock::now();
    deviceKernel<<<32, 256>>>(a, N);
    cudaDeviceSynchronize();
    auto endGPU = std::chrono::high_resolution_clock::now();
    printf("GPU Execution Time: %f ms\n", std::chrono::duration<float, std::milli>(endGPU - startGPU).count());

    // Verify correctness
    for (int i = 0; i < N; ++i)
    {
        if (a[i] != 1)
        {
            printf("Error at index %d: %d\n", i, a[i]);
            break;
        }
    }
    printf("Verification complete.\n");

    cudaFree(a);
    return 0;
}