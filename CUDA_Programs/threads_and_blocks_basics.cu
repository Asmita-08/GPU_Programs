%%writefile thread_hierarchy.cu
#include <stdio.h>

// Kernel to print thread and block IDs
__global__ void printThreadInfo() {
    // Get thread and block indices
    int threadID = threadIdx.x;
    int blockID  = blockIdx.x;
    
    printf("Hello from Block %d, Thread %d\n", blockID, threadID);
}

int main() {
    // Launch kernel with 2 blocks, each containing 4 threads
    printThreadInfo<<<2, 4>>>();

    // Ensure GPU execution completes before exiting
    cudaDeviceSynchronize();
    
    return 0;
}
