# GPU Programming: CUDA & HIP (NVIDIA & AMD)

This repository contains GPU-based parallel programming examples using CUDA and HIP frameworks. The goal is to compare and demonstrate the behavior of GPU programs across different architectures—NVIDIA and AMD—using CUDA and HIP in both CUDA and ROCm environments.

## 🚀 Frameworks Used

- **CUDA (Compute Unified Device Architecture)** – for NVIDIA GPUs  
- **HIP (Heterogeneous-Compute Interface for Portability)**  
  - HIP on **NVIDIA GPU** (compiled with `hipcc` using CUDA backend)  
  - HIP on **AMD GPU** (compiled with `hipcc` using ROCm backend)
 
## ⚙️ Environment Details

- **CUDA Version**: 12.5  
- **HIP Version**: ROCm 6.x (for AMD), HIP with CUDA backend (for NVIDIA)  
- **GPUs Used**:  
  - NVIDIA T1000 8GB  
  - AMD Radeon (ROCm Supported)  MI50/MI60
- **Development Platform**:  
  - Google Colab (CUDA)  
  - ROCm Environment on AMD Server  
  - AlmaLinux 9.5 (Host System)
 
## 📖 How to Run

> Make sure to have the appropriate GPU and drivers installed (CUDA or ROCm).
[The compilation command may vary on the basis of the system being used and CUDA Version, therefore, check it before using]

# To compile and run the CUDA Programs
> To compile the CUDA Programs-
  nvcc -Wno-deprecated-gpu-targets filename.cu -o outputfile 

> To run the CUDA Programs-
  ./outputfile

# To compile and run the HIP Programs
# For NVIDIA (CUDA backend)
> To compile-
  hipcc filename.cpp -o outputfile

> To run-
  ./outputfile 

# For AMD (ROCm backend)
> To compile-
  hipcc filename.cpp -o outputfile

> To run-
  ./outputfile
 
  
# Purpose
This project is developed as part of a Parallel Programming Comparative Study to understand GPU computing frameworks and their behavior across different architectures.


Feel free to raise an issue or reach out if you have any questions or suggestions!
