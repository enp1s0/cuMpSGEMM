# CuPy hijacking test

## Requirements
- numpy
- cupy

## Run
```bash
export LD_PRELOAD=/path/to/libcumpsgemm.so:$LD_PRELOAD
python main.py
```
