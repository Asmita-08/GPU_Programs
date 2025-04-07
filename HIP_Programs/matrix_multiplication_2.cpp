#include <hip/hip_runtime.h>
#include <iostream>

#define N 4  // Matrix size

__global__ void matrix_multiply(int *A, int *B, int *C, int size) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    int sum = 0;
    for (int k = 0; k < size; k++) {
        sum += A[row * size + k] * B[k * size + col];
    }
    C[row * size + col] = sum;
}

int main() {
    int *A, *B, *C;
    int size = N * N * sizeof(int);

    hipMallocManaged(&A, size);
    hipMallocManaged(&B, size);
    hipMallocManaged(&C, size);

    for (int i = 0; i < N * N; i++) {
        A[i] = i + 1;
        B[i] = (i + 1) % N;
    }

    dim3 threadsPerBlock(N, N);
    matrix_multiply<<<1, threadsPerBlock>>>(A, B, C, N);
    hipDeviceSynchronize();

    std::cout << "Result Matrix C:\n";
    for (int i = 0; i < N * N; i++) {
        std::cout << C[i] << " ";
        if ((i + 1) % N == 0) std::cout << "\n";
    }

    hipFree(A);
    hipFree(B);
    hipFree(C);
    return 0;
}

