#include<cuda_runtime.h>
#include <stdio.h>

__global__ void vector_add(int *a, int *b, int *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

int main() {
    int n = 10;
    int h_a[10], h_b[10], h_c[10];
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(int));
    cudaMalloc(&d_b, n * sizeof(int));
    cudaMalloc(&d_c, n * sizeof(int));

    cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);

    vector_add<<<1, n>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++)
        printf("\n %d ", h_c[i]);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}



