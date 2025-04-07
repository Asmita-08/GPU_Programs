#include<stdio.h>
#include<cuda.h>

__global__ void dimension_kernel(){
        if(threadIdx.x==0 && blockIdx.x==0 && threadIdx.y==0 && blockIdx.y==0 && threadIdx.z==0 && blockIdx.z==0){
                printf("\n %d %d %d %d %d %d", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
        }
}

int main(){
        dim3 grid(2, 3, 4);
        dim3 block(5, 6, 7);
        dimension_kernel<<<grid, block>>>();
        cudaDeviceSynchronize();
        return 0;
}

/*Number of threads launched= 2*3*4*5*6*7.
Number of threads in a thread-block= 5*6*7.
Number of thread-blocks in a grid= 2*3*4.*/


