# cuMpSGEMM - CUDA Mutable-precision SGEMM

An SGEMM precision tolerance checking library hijacking cuBLAS SGEMM call.

## Installation
```
git clone https://github.com/enp1s0/cuMpSGEMM.git --recursive
cd cuMpSGEMM
mkdir build
cd build
cmake ..
make -j4
```

- Add `-DCMAKE_INSTALL_PREFIX=/path/to/install` option to `cmake` if you want to specify the installation path and run `make install` after the last `make` command.

## Usage

1. Set `LD_LIBRARY_PATH` and `LIBRARY_PATH`.

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/cuMpSGEMM/build
export LIBRARY_PATH=$LIBRARY_PATH:/path/to/cuMpSGEMM/build

# When the installation path has been specified, execute
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/install
export LIBRARY_PATH=$LIBRARY_PATH:/path/to/install
```

2. Add `-lcumpsgemm` and `-lcuda` to the compiler option **before -lcublas option**, when compiling the target application.

```bash
# Example
nvcc main.cu ... -lcumpsgemm -lcuda ... -lcublas ...
```

3. Run the execution file as usual.

## License
MIT
