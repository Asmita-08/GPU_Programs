#include <hip/hip_runtime.h>
#include <stdio.h>

// Simulated hardware values
#define NUM_SMS 14         // NVIDIA SMs or AMD Compute Units
#define WARP_SIZE 32       // 32 (NVIDIA), could be 64 on AMD, kept 32 for uniformity

// HIP Kernel
__global__ void threadMappingKernelHIP() {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int blockDimX = blockDim.x;
    int blockDimY = blockDim.y;
    int blockDimZ = blockDim.z;

    // Flatten thread ID within block
    int localThreadId = (tz * blockDimY * blockDimX) + (ty * blockDimX) + tx;

    int threadsPerBlock = blockDimX * blockDimY * blockDimZ;
    int blockId = (bz * gridDim.y * gridDim.x) + (by * gridDim.x) + bx;
    int globalThreadId = blockId * threadsPerBlock + localThreadId;

    // Warp and SM ID Simulation
    int warpId = globalThreadId / WARP_SIZE;
    int simulatedSmId = blockId % NUM_SMS;

    printf("Block(%d,%d,%d) Thread(%d,%d,%d) → Global Thread ID: %d, Warp ID: %d, Simulated SM ID: %d\n",
           bx, by, bz, tx, ty, tz, globalThreadId, warpId, simulatedSmId);
}

int main() {
    // Set grid and block dimensions
    dim3 gridDim(2, 2, 1);     // 4 blocks
    dim3 blockDim(8, 8, 2);    // 128 threads per block

    int totalThreads = gridDim.x * gridDim.y * gridDim.z *
                       blockDim.x * blockDim.y * blockDim.z;

    printf("Launching HIP kernel with %d blocks and %d threads total...\n\n",
           gridDim.x * gridDim.y * gridDim.z, totalThreads);

    hipLaunchKernelGGL(threadMappingKernelHIP, gridDim, blockDim, 0, 0);
    hipDeviceSynchronize();

    return 0;
}


