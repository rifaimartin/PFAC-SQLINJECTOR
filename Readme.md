# PFAC-SQLINJECTOR

A high-performance SQL injection detection tool using parallel pattern matching algorithms on CUDA.

## Overview

PFAC-SQLINJECTOR utilizes the Parallel Failure-less Aho-Corasick (PFAC) algorithm implemented on NVIDIA CUDA to efficiently detect potential SQL injection patterns in large query datasets. The project employs GPU acceleration to significantly improve the processing speed of query validation compared to traditional CPU-based methods.

## Requirements

- NVIDIA CUDA Toolkit (12.0 or newer recommended)
- NVIDIA GPU with Compute Capability 3.5 or higher
- C++ compiler with C++11 support

## Cuda Implementation

```bash
# Clone the repository
git clone https://github.com/yourusername/PFAC-SQLINJECTOR.git
cd PFAC-SQLINJECTOR

# Compile with NVCC (NVIDIA CUDA Compiler)
nvcc -o Aho-Corasick Aho-Corasick.cu

nvcc -o PFAC PFAC.cu
```

## Algorithm Details


## Performance

Performance metrics on a test dataset:

| Dataset Size | Processing Time | Queries/Second |
|-------------|-----------------|---------------|
| - | - | - |
| - | - | - |
| - | - | - |

*Tested on NVIDIA GeForce GTX 1660 Ti (benchmarks may vary based on hardware)*

## Project Structure


## SQL Injection Pattern Categories


## Risk Classification System

- **Low Risk (≤30)**: Potential SQL keywords detected but likely benign
- **Medium Risk (31-70)**: Multiple SQL patterns detected that might indicate an attack
- **High Risk (71-90)**: Clear evidence of SQL manipulation attempts
- **Critical Risk (>90)**: Destructive operations or command execution attempts detected

## Authors

- Robertus Hudi - Universitas Pelita Harapan
- Kennedy Suganto - Universitas Pelita Harapan
- Muhammad Rifai - Universitas Pelita Harapan

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This research was supported by Universitas Pelita Harapan and the Institute of Research and Community Services (LPPM UPH) with research number P-92-SISTech-VII/2023.
