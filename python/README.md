# Python API for controlling cuBLAS hijacking

## Installation
```
python setup.py install
# or
pip install .
```

## Usage
Before using this Python package, please confirm `LD_LIBRARY_PATH` or `LD_PRELOAD` is set.
```python
import cumpsgemm_hijack_control as chc

# Check whether the cuBLAS library is successfully hijacked
print(chc.is_library_loaded())

# Set the computing mode to CUMPSGEMM_FP16TCEC
chc.set_compute_mode(chc.CUMPSGEMM_FP16TCEC)

# Enable the fast (M, 2, 2) and (2, N, 2) GEMM
chc.enable_custom_gemm_Mx2x2()
#chc.disable_custom_gemm_Mx2x2()

# AUTO mode configuration
## Set (ignore_threshold, underflow_threshold, underflow_tolerance_rate)
chc.set_exp_stats_params(1e-15, 1e-5, 0.2)

## Disable restoring A and B scaling after the GEMM computation.
## If the matrices are not used for another computation, it should be disabled
chc.disable_restoring_AB_after_scaling()
#chc.enable_restoring_AB_after_scaling()

# cupy.matmul(a, b) ...

# Unset the computing mode and use the default rule
chc.unset_compute_mode()
```
