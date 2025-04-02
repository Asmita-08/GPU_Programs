#include <hip/hip_runtime.h>
#include <stdio.h>

#define N 16

__global__ void mat_mult(int *A, int *B, int *C, int n) {
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

int main() {
    int *d_A, *d_B, *d_C;
    hipMalloc(&d_A, N * N * sizeof(int));
    hipMalloc(&d_B, N * N * sizeof(int));
    hipMalloc(&d_C, N * N * sizeof(int));

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / 16, N / 16);
    mat_mult<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    return 0;
}
