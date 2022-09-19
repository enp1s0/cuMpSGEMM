# Python API for controlling cuBLAS hijacking

## Installation
```
python setup.py install
# or
pip install .
```

## Usage
```python
import cumpsgemm_hijack_control as chc

# To set the computing mode to CUMPSGEMM_FP16TCEC
chc.set_compute_mode(chc.CUMPSGEMM_FP16TCEC)

# To unset the computing mode and use the default rule
chc.unset_compute_mode()
```
