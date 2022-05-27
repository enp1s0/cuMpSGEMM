# CuPy hijacking test

## Requirements
- numpy
- cupy

## Run
```bash
export LD_LIBRARY_PATH=/path/to/cumpsgemm/hijack/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/path/to/libcumpsgemm.so
python main.py
```
