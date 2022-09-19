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

# To set the computing mode to CUMPSGEMM_FP16TCEC
chc.set_compute_mode(chc.CUMPSGEMM_FP16TCEC)
# cupy.matmul(a, b) ...

# To unset the computing mode and use the default rule
chc.unset_compute_mode()
```
