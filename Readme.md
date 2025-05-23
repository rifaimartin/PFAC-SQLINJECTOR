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

# Create Nsys-Report
nsys profile -o Aho-Corasick-[Level_SQL]-[LengthDatSize] --stats=true ./Aho-Corasick.exe

nvcc -o PFAC PFAC.cu

# Run with dataset[size]
PFAC.exe -d sql_dataset_Low_5000

Aho-Corasick -d sql-dataset_Low_5000


```

# Untuk PFAC
./PFAC.exe -d sql_dataset_Low_5000

# Untuk Aho-Corasick
./Aho-Corrasick.exe -d sql_dataset_High_1000

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


Algorithm,Dataset_Size,Pattern_Count,Execution_Time_μs,Risk_Level
PFAC,10,72,69.403,Low
PFAC,100,72,68.403,Low
PFAC,1000,72,73.560,Low
PFAC,5000,72,60.469,Low
PFAC,10000,72,68.652,Low

PFAC,10,72,68.589,Medium
PFAC,100,72,67.872,Medium
PFAC,1000,72,77.125,Medium
PFAC,5000,72,68.651,Medium
PFAC,10000,72,54.712,Medium


PFAC,10,72,74.639,High
PFAC,100,72,77.411,High
PFAC,1000,72,74.006,High
PFAC,5000,72,66.497,High
PFAC,10000,72,57.793,High

PFAC,10,72,67.157,Critical
PFAC,100,72,64.002,Critical
PFAC,1000,72,70.865,Critical
PFAC,5000,72,58.581,Critical
PFAC,10000,72,60.385,Critical



Aho-Corasick,10,72,105.094,Low
Aho-Corasick,100,72,57.555,Low
Aho-Corasick,1000,72,64.729,Low
Aho-Corasick,5000,72,66.546,Low
Aho-Corasick,10000,72,77.097,Low

Aho-Corasick,10,72,53.738,Medium
Aho-Corasick,100,72,66.980,Medium
Aho-Corasick,1000,72,53.138,Medium
Aho-Corasick,5000,72,79.562,Medium
Aho-Corasick,10000,72,87.640,Medium

Aho-Corasick,10,72,58.752,High
Aho-Corasick,100,72,55.067,High
Aho-Corasick,1000,72,92.564,High
Aho-Corasick,5000,72,61.689,High
Aho-Corasick,10000,72,81.773,High


Aho-Corasick,10,72,65.693,Critical
Aho-Corasick,100,72,75.002,Critical
Aho-Corasick,1000,72,66.528,Critical
Aho-Corasick,5000,72,85.181,Critical
Aho-Corasick,10000,72,82.860,Critical


