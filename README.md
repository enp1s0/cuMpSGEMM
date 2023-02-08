# cuMpSGEMM - CUDA Mutable-precision SGEMM

An SGEMM precision tolerance checking library hijacking cuBLAS SGEMM function call.

**Note**

This library is only for checking SGEMM precision tolerance.
The computing throughput is low since we don't optimize the kernel function.

## Supported functions
- `cublasSgemm`
- `cublasCgemm`
- `cublasGemmEx` (Only for single precision)

## Installation
- For A100 (GA100) GPU

```
git clone https://github.com/enp1s0/cuMpSGEMM.git --recursive
cd cuMpSGEMM
mkdir build
cd build
cmake ..
# It may take ~15 min
make -j4
```

- For other Ampere GPUs

Comment out the following line in `src/handle.cu` before `make`.
```
#define ENABLE_A100_OPTIMAZED_PARAMETERS
```
Without this modification, an error may occur in dynamic shared memory size configuration step in runtime.
The GA100 architecture has more shared memory size than other Ampere GPUs and the optimization for GA100 is besed on the shared memory size.
This throughput of this library is only optimized for A100 GPU.
(We have used A100 40GB SXM4 for the prameters optimization.)

## Usage

### 1. Hijack cuBLAS library
Before executing a target application, set an environmental variable as follows.
```bash
export LD_PRELOAD=/path/to/cumpsgemm/hijack/lib/libcumpsgemm.so:$LD_PRELOAD
```

### 2. Control SGEMM computing mode
By the default rule, the SGEMM computing mode can be changed via an environmental variable as follows:

```bash
export CUMPSGEMM_COMPUTE_MODE=FP16TCEC
```

| mode name            | Tensor Core Type               | Error Correction |
|:---------------------|:-------------------------------|:-----------------|
|`FP16TCEC`            | FP16                           | Yes              |
|`TF32TCEC`            | TF32                           | Yes              |
|`FP16TC`              | FP16                           | No               |
|`TF32TC`              | TF32                           | No               |
|`CUBLAS`              | Depends on the cublas math mode| No               |
|`CUBLAS_SIMT`         | (FP32 SIMT Core)               | No               |
|`CUBLAS_FP16`         | FP16                           | No               |
|`CUBLAS_TF32`         | TF32                           | No               |
|`AUTO`                | AUTO                           | Yes              |
|`FP16TCEC_SCALING`    | FP16                           | Yes              |
|`DRY_RUN`             | Nothing is computed            | No               |

#### Custom rule
By defining a custom `cuMpSGEMM_get_compute_mode` function and including it in a shared library named `libcumpsgemm_rule.so`, the SGEMM mode can be changed as you want.
The default function definition is in [default_cumpsgemm_rule.cu](src/default_cumpsgemm_rule.cu).
Before executing a target application, set an environmental variable as follows.
```bash
export LD_LIBRARY_PATH=/path/to/libcumpsgemm_rule.so/dir:$LD_LIBRARY_PATH
```

## How this library works

![cuMpSGEMM flow](./docs/cumpsgemm.svg)

When a supported cuBLAS function (e.g. `cublasSgemm`) is called, a function selector inside this library calls `cuMpSGEMM_get_compute_mode` function (1) to determine the backend SGEMM function (2).
Then it calls an appropriate function (3).

## Important note
To hijack the cuBLAS static library, the same name library is created.
In this process, the build script decomposes the cuBLAS static library and composes the TCEC SGEMM and decomposed modules except sgemm.o etc.
This is not the reverse engineering, decompiling or disassembling that is prohibited by [NVIDIA EULA](https://docs.nvidia.com/cuda/eula/index.html).

## Test
```
Usage : ./build/cumpsgemm_test sgemm [exp2|seq] [min_N] [max_N] [interval]
      : ./build/cumpsgemm_test cgemm [exp2|seq] [min_N] [max_N] [interval]
      : ./build/cumpsgemm_test sgemm_strided_batch [exp2|seq] [min_N] [max_N] [interval] [batch_count]
      : ./build/cumpsgemm_test cgemm_strided_batch [exp2|seq] [min_N] [max_N] [interval] [batch_count]
      : ./build/cumpsgemm_test cublas_sgemm [exp2|seq] [min_N] [max_N] [interval]
      : ./build/cumpsgemm_test cublas_cgemm [exp2|seq] [min_N] [max_N] [interval]
      : ./build/cumpsgemm_test cublas_sgemm_strided_batch [exp2|seq] [min_N] [max_N] [interval] [batch_count]
      : ./build/cumpsgemm_test cublas_cgemm_strided_batch [exp2|seq] [min_N] [max_N] [interval] [batch_count]
      : ./build/cumpsgemm_test log [/path/to/log]
```

## Controling environmental variables
```bash
# Select a GEMM implementation executing
export CUMPSGEMM_COMPUTE_MODE=[FP16TC|FP16TCEC|TF32TC|TF32TCEC|CUBLAS]

# Output debug information
export CUMPSGEMM_INFO=[0|1]

# Output error message
export CUMPSGEMM_ERROR_LOG=[0|1]

# Enable custom gemm_Mx2x2 (https://github.com/enp1s0/cuGEMM-Mx2x2)
export CUMPSGEMM_CUSTOM_GEMM_MX2X2=[0|1]
```

### CULiP integration
To output [CULiP](https://github.com/enp1s0/CULiP) logs, specify a following environmental variable.
```bash
export CUMPSGEMM_ENABLE_CULIP_PROFILING=1
```

## License
MIT
