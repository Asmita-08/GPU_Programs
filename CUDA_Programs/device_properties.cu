#include<cuda_runtime.h>
#include <stdio.h>

int main()
{
  //Device ID is required first to query the device.

  int deviceId;
  cudaGetDevice(&deviceId);

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);

  //The following contains various properties about the current device.

  int computeCapabilityMajor = props.major;
  int computeCapabilityMinor = props.minor;
  int multiProcessorCount = props.multiProcessorCount;
  int warpSize = props.warpSize;


  printf("Device ID: %d\nNumber of SMs: %d\nCompute Capability Major: %d\nCompute Capability Minor: %d\nWarp Size: %d\n", deviceId, multiProcessorCount, computeCapabilityMajor, computeCapabilityMinor, warpSize);
}