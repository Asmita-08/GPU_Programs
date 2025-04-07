#include<stdio.h>
#include<cuda.h>

__global__ void myKernel(char *arr, int l){
        unsigned id = threadIdx.x;
        if(id<l){
                ++arr[id];
        }
}

int main(){
        char cpuarr[]="abcdefgh12345678";
        char *gpuarr;

        cudaMalloc(&gpuarr, sizeof(char) *(1 + strlen(cpuarr)));

        cudaMemcpy(gpuarr, cpuarr, sizeof(char) *(1 + strlen(cpuarr)), cudaMemcpyHostToDevice);

        myKernel<<<1, 32>>>(gpuarr, strlen(cpuarr) + 1);

        cudaDeviceSynchronize();

	cudaMemcpy(cpuarr, gpuarr, sizeof(char) *(1 + strlen(cpuarr)), cudaMemcpyDeviceToHost);

        printf(cpuarr);
        return 0;
}


