#include <hip/hip_runtime.h>
#include <iostream>

#define N 1024  // Array size
#define BINS 10 // Number of bins in the histogram

__global__ void histogram(int *arr, int *hist) {
    __shared__ int local_hist[BINS];
    int tid = threadIdx.x;

    if (tid < BINS)
        local_hist[tid] = 0;  // Initialize shared memory
    __syncthreads();

    atomicAdd(&local_hist[arr[tid] % BINS], 1);  // Count occurrences
    __syncthreads();

    if (tid < BINS)
        atomicAdd(&hist[tid], local_hist[tid]);  // Store final result
}

int main() {
    int *arr, *hist;

    hipMallocManaged(&arr, N * sizeof(int));
    hipMallocManaged(&hist, BINS * sizeof(int));

    for (int i = 0; i < N; i++) arr[i] = i % BINS;  // Generate numbers between 0-9

    histogram<<<1, N>>>(arr, hist);
    hipDeviceSynchronize();

    std::cout << "Histogram:\n";
    for (int i = 0; i < BINS; i++)
        std::cout << "Bin " << i << ": " << hist[i] << std::endl;

    hipFree(arr);
    hipFree(hist);
    return 0;
}


