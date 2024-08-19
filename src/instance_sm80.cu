#include "cumpsgemm_kernel.cuh"
#include "handle.hpp"
#include "instance.hpp"

void cumpsgemm::configure_instance_sm80(
    cumpsgemm::gemm_module gemm_module[cumpsgemm::kernel_module_code::max_code]
                                      [cumpsgemm::num_kernel_candidates],
    cumpsgemm::gemm_module
        gemm_stridedBatch_module[cumpsgemm::kernel_module_code::max_code]
                                [cumpsgemm::num_kernel_candidates],
    cumpsgemm::gemm_module
        gemm_atomic_module[cumpsgemm::kernel_module_code::max_code]) {
  using tf32 = nvcuda::wmma::precision::tf32;
#ifdef COMPILE_SGEMM_KERNEL
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, col_major,
                         col_major, 64, 128, 32, 32, 64, 32, 128, 1, 2, false,
                         s, 0); // N=  16384, p= 47.33 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, col_major,
                         col_major, 64, 128, 32, 32, 64, 32, 128, 1, 2, false,
                         s, 1); // N=   4096, p= 46.33 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, col_major,
                         col_major, 64, 128, 32, 32, 64, 32, 128, 1, 2, false,
                         s, 2); // N=   1024, p= 21.54 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, col_major,
                         col_major, 64, 128, 32, 32, 64, 16, 128, 1, 2, false,
                         s, 0); // N=  16384, p= 29.71 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, col_major,
                         col_major, 64, 128, 32, 64, 32, 16, 128, 1, 2, false,
                         s, 1); // N=   4096, p= 29.51 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, col_major,
                         col_major, 64, 64, 32, 32, 32, 16, 128, 1, 2, false, s,
                         2); // N=   1024, p= 20.13 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, col_major,
                         col_major, 128, 128, 32, 64, 64, 16, 128, 1, 2, false,
                         s, 0); // N=  16384, p= 75.69 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, col_major,
                         col_major, 128, 128, 32, 64, 64, 32, 128, 1, 2, false,
                         s, 1); // N=   4096, p= 87.86 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, col_major,
                         col_major, 128, 128, 64, 64, 64, 32, 128, 2, 2, false,
                         s, 2); // N=   1024, p= 47.52 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, col_major,
                         col_major, 64, 128, 32, 64, 32, 16, 128, 2, 2, false,
                         s, 0); // N=  16384, p= 64.93 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, col_major,
                         col_major, 128, 128, 32, 64, 64, 32, 128, 2, 2, false,
                         s, 1); // N=   4096, p= 69.67 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, col_major,
                         col_major, 128, 128, 32, 64, 64, 32, 128, 2, 2, false,
                         s, 2); // N=   1024, p= 37.49 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, col_major,
                         row_major, 128, 64, 32, 64, 32, 32, 128, 1, 2, false,
                         s, 0); // N=  16384, p= 39.64 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, col_major,
                         row_major, 128, 64, 32, 64, 32, 32, 128, 1, 2, false,
                         s, 1); // N=   4096, p= 40.81 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, col_major,
                         row_major, 128, 32, 32, 32, 32, 16, 128, 2, 2, false,
                         s, 2); // N=   1024, p= 24.70 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, col_major,
                         row_major, 64, 128, 32, 32, 64, 16, 128, 1, 2, false,
                         s, 0); // N=  16384, p= 30.35 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, col_major,
                         row_major, 64, 128, 32, 32, 64, 16, 128, 1, 2, false,
                         s, 1); // N=   4096, p= 30.38 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, col_major,
                         row_major, 32, 128, 32, 32, 32, 16, 128, 1, 2, false,
                         s, 2); // N=   1024, p= 18.67 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, col_major,
                         row_major, 128, 128, 32, 64, 64, 32, 128, 1, 2, false,
                         s, 0); // N=  16384, p= 69.42 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, col_major,
                         row_major, 128, 128, 32, 64, 64, 32, 128, 1, 2, false,
                         s, 1); // N=   4096, p= 76.70 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, col_major,
                         row_major, 128, 128, 64, 64, 64, 64, 128, 1, 2, false,
                         s, 2); // N=   1024, p= 43.22 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, col_major,
                         row_major, 128, 128, 32, 32, 64, 16, 256, 1, 2, false,
                         s, 0); // N=  16384, p= 65.32 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, col_major,
                         row_major, 128, 64, 32, 64, 32, 32, 128, 2, 2, false,
                         s, 1); // N=   4096, p= 71.81 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, col_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, s,
                         2); // N=   1024, p= 41.37 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, row_major,
                         col_major, 64, 128, 32, 64, 32, 32, 128, 1, 2, false,
                         s, 0); // N=  16384, p= 48.66 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, row_major,
                         col_major, 64, 128, 32, 32, 64, 32, 128, 1, 2, false,
                         s, 1); // N=   4096, p= 51.17 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, row_major,
                         col_major, 64, 64, 64, 32, 32, 64, 128, 1, 2, false, s,
                         2); // N=   1024, p= 27.83 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, row_major,
                         col_major, 64, 128, 32, 64, 32, 16, 128, 1, 2, false,
                         s, 0); // N=  16384, p= 30.00 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, row_major,
                         col_major, 64, 128, 32, 64, 32, 16, 128, 1, 2, false,
                         s, 1); // N=   4096, p= 29.82 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, row_major,
                         col_major, 64, 64, 32, 32, 32, 16, 128, 1, 2, false, s,
                         2); // N=   1024, p= 18.83 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, row_major,
                         col_major, 64, 128, 32, 32, 64, 16, 128, 1, 2, false,
                         s, 0); // N=  16384, p= 73.09 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, row_major,
                         col_major, 128, 128, 32, 64, 64, 32, 128, 1, 2, false,
                         s, 1); // N=   4096, p= 95.97 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, row_major,
                         col_major, 128, 128, 64, 64, 64, 32, 128, 1, 2, false,
                         s, 2); // N=   1024, p= 51.86 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, row_major,
                         col_major, 64, 128, 32, 32, 64, 32, 128, 2, 2, false,
                         s, 0); // N=  16384, p= 61.45 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, row_major,
                         col_major, 128, 128, 32, 64, 64, 32, 128, 2, 2, false,
                         s, 1); // N=   4096, p= 66.13 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, row_major,
                         col_major, 128, 128, 32, 64, 64, 32, 128, 1, 2, false,
                         s, 2); // N=   1024, p= 34.48 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, row_major,
                         row_major, 64, 128, 32, 64, 32, 32, 128, 1, 2, false,
                         s, 0); // N=  16384, p= 46.80 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, row_major,
                         row_major, 64, 128, 32, 64, 32, 32, 128, 1, 2, false,
                         s, 1); // N=   4096, p= 48.76 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, row_major,
                         row_major, 128, 32, 32, 32, 32, 16, 128, 1, 2, false,
                         s, 2); // N=   1024, p= 26.49 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, row_major,
                         row_major, 64, 128, 32, 32, 64, 16, 128, 1, 2, false,
                         s, 0); // N=  16384, p= 30.49 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, row_major,
                         row_major, 64, 128, 32, 32, 64, 16, 128, 1, 2, false,
                         s, 1); // N=   4096, p= 30.29 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, row_major,
                         row_major, 32, 128, 32, 32, 32, 16, 128, 1, 2, false,
                         s, 2); // N=   1024, p= 18.95 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, row_major,
                         row_major, 64, 128, 32, 64, 32, 32, 128, 1, 2, false,
                         s, 0); // N=  16384, p= 68.53 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, row_major,
                         row_major, 128, 128, 32, 64, 64, 32, 128, 1, 2, false,
                         s, 1); // N=   4096, p= 83.53 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, row_major,
                         row_major, 128, 128, 64, 64, 64, 32, 128, 1, 2, false,
                         s, 2); // N=   1024, p= 48.39 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, row_major,
                         row_major, 64, 128, 32, 32, 64, 32, 128, 1, 2, false,
                         s, 0); // N=  16384, p= 64.50 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, row_major,
                         row_major, 64, 128, 32, 32, 64, 32, 128, 2, 2, false,
                         s, 1); // N=   4096, p= 69.98 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, row_major,
                         row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
                         2); // N=   1024, p= 38.56 [TFlop/s]
#endif
#ifdef COMPILE_CGEMM_KERNEL
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, col_major,
                         col_major, 128, 64, 32, 32, 32, 16, 256, 1, 2, false,
                         c, 0); // N=   8192, p= 46.01 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, col_major,
                         col_major, 64, 128, 32, 16, 64, 16, 256, 2, 2, false,
                         c, 1); // N=   2048, p= 33.33 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, col_major,
                         col_major, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 17.78 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, col_major,
                         col_major, 64, 128, 32, 16, 64, 16, 256, 1, 2, false,
                         c, 0); // N=   8192, p= 31.35 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, col_major,
                         col_major, 64, 64, 32, 16, 64, 16, 128, 2, 2, false, c,
                         1); // N=   2048, p= 30.10 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, col_major,
                         col_major, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 18.28 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, col_major,
                         col_major, 128, 64, 32, 32, 64, 16, 128, 2, 2, false,
                         c, 0); // N=   8192, p= 96.94 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, col_major,
                         col_major, 128, 64, 32, 32, 64, 16, 128, 1, 2, false,
                         c, 1); // N=   2048, p= 92.30 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, col_major,
                         col_major, 64, 64, 64, 16, 64, 32, 128, 2, 2, false, c,
                         2); // N=    512, p= 34.33 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, col_major,
                         col_major, 128, 64, 32, 32, 32, 16, 256, 1, 2, false,
                         c, 0); // N=   8192, p= 66.51 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, col_major,
                         col_major, 128, 64, 32, 32, 32, 16, 256, 1, 2, false,
                         c, 1); // N=   2048, p= 62.94 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, col_major,
                         col_major, 32, 32, 32, 16, 16, 32, 128, 1, 2, false, c,
                         2); // N=    512, p= 29.02 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, col_major,
                         row_major, 64, 64, 32, 16, 64, 16, 128, 1, 2, false, c,
                         0); // N=   8192, p= 40.78 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, col_major,
                         row_major, 64, 64, 32, 16, 64, 16, 128, 1, 2, false, c,
                         1); // N=   2048, p= 40.68 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, col_major,
                         row_major, 32, 32, 32, 16, 16, 32, 128, 1, 2, false, c,
                         2); // N=    512, p= 19.13 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, col_major,
                         row_major, 128, 64, 32, 16, 64, 16, 256, 1, 2, false,
                         c, 0); // N=   8192, p= 31.21 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, col_major,
                         row_major, 64, 64, 32, 16, 64, 16, 128, 1, 2, false, c,
                         1); // N=   2048, p= 29.72 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, col_major,
                         row_major, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 16.74 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, col_major,
                         row_major, 128, 128, 32, 32, 64, 16, 256, 1, 2, false,
                         c, 0); // N=   8192, p= 73.16 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, col_major,
                         row_major, 64, 128, 32, 32, 64, 16, 128, 1, 2, false,
                         c, 1); // N=   2048, p= 67.79 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, col_major,
                         row_major, 64, 64, 64, 32, 32, 32, 128, 1, 2, false, c,
                         2); // N=    512, p= 26.70 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, col_major,
                         row_major, 128, 64, 32, 32, 32, 16, 256, 1, 2, false,
                         c, 0); // N=   8192, p= 66.45 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, col_major,
                         row_major, 128, 64, 32, 32, 32, 16, 256, 1, 2, false,
                         c, 1); // N=   2048, p= 61.95 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, col_major,
                         row_major, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 27.80 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, col_major,
                         conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
                         0); // N=   8192, p= 39.59 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, col_major,
                         conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
                         1); // N=   2048, p= 35.83 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, col_major,
                         conjugate, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 17.04 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, col_major,
                         conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
                         0); // N=   8192, p= 28.55 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, col_major,
                         conjugate, 64, 32, 32, 16, 32, 32, 128, 1, 2, false, c,
                         1); // N=   2048, p= 27.16 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, col_major,
                         conjugate, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 15.40 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, col_major,
                         conjugate, 128, 128, 32, 32, 64, 16, 256, 2, 2, false,
                         c, 0); // N=   8192, p= 65.10 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, col_major,
                         conjugate, 128, 64, 32, 32, 64, 16, 128, 1, 2, false,
                         c, 1); // N=   2048, p= 49.61 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, col_major,
                         conjugate, 128, 32, 32, 32, 32, 16, 128, 2, 2, false,
                         c, 2); // N=    512, p= 20.20 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, col_major,
                         conjugate, 128, 128, 32, 32, 64, 16, 256, 1, 2, false,
                         c, 0); // N=   8192, p= 55.12 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, col_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // N=   2048, p= 49.27 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, col_major,
                         conjugate, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 22.58 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, row_major,
                         col_major, 64, 128, 32, 32, 32, 16, 256, 2, 2, false,
                         c, 0); // N=   8192, p= 53.69 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, row_major,
                         col_major, 128, 64, 32, 32, 32, 16, 256, 1, 2, false,
                         c, 1); // N=   2048, p= 51.44 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, row_major,
                         col_major, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 26.57 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, row_major,
                         col_major, 128, 64, 32, 16, 64, 16, 256, 1, 2, false,
                         c, 0); // N=   8192, p= 31.06 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, row_major,
                         col_major, 64, 64, 32, 16, 64, 16, 128, 1, 2, false, c,
                         1); // N=   2048, p= 30.28 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, row_major,
                         col_major, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 17.62 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, row_major,
                         col_major, 64, 128, 32, 32, 64, 16, 128, 1, 2, false,
                         c, 0); // N=   8192, p=119.95 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, row_major,
                         col_major, 128, 64, 32, 64, 32, 16, 128, 1, 2, false,
                         c, 1); // N=   2048, p=120.93 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, row_major,
                         col_major, 32, 64, 32, 32, 16, 16, 128, 2, 2, false, c,
                         2); // N=    512, p= 43.71 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, row_major,
                         col_major, 128, 128, 32, 32, 64, 16, 256, 1, 2, false,
                         c, 0); // N=   8192, p= 65.12 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, row_major,
                         col_major, 128, 64, 32, 32, 32, 16, 256, 1, 2, false,
                         c, 1); // N=   2048, p= 60.01 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, row_major,
                         col_major, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 28.26 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, row_major,
                         row_major, 64, 128, 32, 32, 32, 16, 256, 2, 2, false,
                         c, 0); // N=   8192, p= 51.36 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, row_major,
                         row_major, 64, 128, 32, 32, 32, 16, 256, 2, 2, false,
                         c, 1); // N=   2048, p= 48.21 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, row_major,
                         row_major, 64, 64, 64, 32, 32, 16, 128, 2, 2, false, c,
                         2); // N=    512, p= 22.50 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, row_major,
                         row_major, 128, 64, 32, 16, 64, 16, 256, 1, 2, false,
                         c, 0); // N=   8192, p= 31.01 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, row_major,
                         row_major, 64, 64, 32, 16, 64, 16, 128, 1, 2, false, c,
                         1); // N=   2048, p= 30.05 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, row_major,
                         row_major, 32, 32, 32, 16, 16, 16, 128, 2, 2, false, c,
                         2); // N=    512, p= 16.93 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, row_major,
                         row_major, 64, 128, 32, 64, 32, 32, 128, 2, 2, false,
                         c, 0); // N=   8192, p=101.51 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, row_major,
                         row_major, 64, 128, 32, 64, 32, 16, 128, 2, 2, false,
                         c, 1); // N=   2048, p= 94.37 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, row_major,
                         row_major, 64, 64, 64, 64, 16, 32, 128, 2, 2, false, c,
                         2); // N=    512, p= 36.75 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, row_major,
                         row_major, 64, 128, 32, 32, 32, 16, 256, 1, 2, false,
                         c, 0); // N=   8192, p= 65.21 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, row_major,
                         row_major, 64, 128, 32, 32, 32, 16, 256, 1, 2, false,
                         c, 1); // N=   2048, p= 60.19 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, row_major,
                         row_major, 32, 32, 32, 16, 16, 32, 128, 1, 2, false, c,
                         2); // N=    512, p= 28.83 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, row_major,
                         conjugate, 64, 64, 32, 32, 32, 16, 128, 1, 2, false, c,
                         0); // N=   8192, p= 44.04 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, row_major,
                         conjugate, 64, 64, 32, 64, 16, 16, 128, 1, 2, false, c,
                         1); // N=   2048, p= 41.01 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, row_major,
                         conjugate, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 18.45 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, row_major,
                         conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
                         0); // N=   8192, p= 28.44 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, row_major,
                         conjugate, 64, 32, 32, 16, 32, 32, 128, 1, 2, false, c,
                         1); // N=   2048, p= 27.14 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, row_major,
                         conjugate, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 15.19 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, row_major,
                         conjugate, 128, 128, 32, 64, 32, 16, 256, 2, 2, false,
                         c, 0); // N=   8192, p= 87.88 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, row_major,
                         conjugate, 128, 64, 32, 64, 16, 16, 256, 1, 2, false,
                         c, 1); // N=   2048, p= 61.03 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, row_major,
                         conjugate, 128, 32, 32, 32, 16, 16, 256, 2, 2, false,
                         c, 2); // N=    512, p= 26.25 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, row_major,
                         conjugate, 128, 128, 32, 32, 64, 16, 256, 2, 2, false,
                         c, 0); // N=   8192, p= 53.60 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, row_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 2, 2, false, c,
                         1); // N=   2048, p= 49.07 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, row_major,
                         conjugate, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 22.12 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, conjugate,
                         col_major, 64, 64, 32, 64, 16, 16, 128, 1, 2, false, c,
                         0); // N=   8192, p= 48.00 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, conjugate,
                         col_major, 64, 64, 32, 32, 32, 16, 128, 1, 2, false, c,
                         1); // N=   2048, p= 43.61 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, conjugate,
                         col_major, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 21.35 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, conjugate,
                         col_major, 64, 64, 32, 16, 64, 16, 128, 1, 2, false, c,
                         0); // N=   8192, p= 29.65 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, conjugate,
                         col_major, 64, 64, 32, 16, 64, 16, 128, 1, 2, false, c,
                         1); // N=   2048, p= 27.16 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, conjugate,
                         col_major, 32, 32, 32, 16, 16, 16, 128, 2, 2, false, c,
                         2); // N=    512, p= 15.83 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, conjugate,
                         col_major, 128, 128, 32, 32, 64, 16, 256, 2, 2, false,
                         c, 0); // N=   8192, p=112.07 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, conjugate,
                         col_major, 64, 128, 32, 32, 32, 16, 256, 2, 2, false,
                         c, 1); // N=   2048, p= 74.09 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, conjugate,
                         col_major, 32, 128, 32, 32, 16, 16, 256, 1, 2, false,
                         c, 2); // N=    512, p= 32.85 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, conjugate,
                         col_major, 128, 128, 32, 32, 64, 16, 256, 1, 2, false,
                         c, 0); // N=   8192, p= 61.07 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, conjugate,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // N=   2048, p= 50.24 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, conjugate,
                         col_major, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 24.56 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, conjugate,
                         row_major, 64, 64, 32, 32, 32, 16, 128, 1, 2, false, c,
                         0); // N=   8192, p= 46.15 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, conjugate,
                         row_major, 64, 64, 32, 64, 16, 16, 128, 2, 2, false, c,
                         1); // N=   2048, p= 42.13 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, conjugate,
                         row_major, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 19.72 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, conjugate,
                         row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
                         0); // N=   8192, p= 29.25 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, conjugate,
                         row_major, 64, 64, 32, 16, 64, 16, 128, 2, 2, false, c,
                         1); // N=   2048, p= 26.90 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, conjugate,
                         row_major, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 15.98 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, conjugate,
                         row_major, 128, 128, 32, 64, 32, 16, 256, 1, 2, false,
                         c, 0); // N=   8192, p= 91.11 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, conjugate,
                         row_major, 64, 64, 32, 64, 16, 16, 128, 2, 2, false, c,
                         1); // N=   2048, p= 65.06 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, conjugate,
                         row_major, 32, 128, 32, 32, 16, 16, 256, 1, 2, false,
                         c, 2); // N=    512, p= 26.82 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, conjugate,
                         row_major, 128, 128, 32, 32, 64, 16, 256, 1, 2, false,
                         c, 0); // N=   8192, p= 61.49 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, conjugate,
                         row_major, 64, 64, 32, 32, 32, 16, 128, 1, 2, false, c,
                         1); // N=   2048, p= 50.85 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, conjugate,
                         row_major, 32, 32, 32, 16, 16, 32, 128, 1, 2, false, c,
                         2); // N=    512, p= 23.08 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, conjugate,
                         conjugate, 64, 64, 32, 64, 16, 16, 128, 1, 2, false, c,
                         0); // N=   8192, p= 42.83 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, conjugate,
                         conjugate, 64, 64, 32, 64, 16, 16, 128, 1, 2, false, c,
                         1); // N=   2048, p= 39.64 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, conjugate,
                         conjugate, 32, 32, 32, 16, 16, 32, 128, 1, 2, false, c,
                         2); // N=    512, p= 17.38 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, conjugate,
                         conjugate, 64, 64, 32, 32, 32, 16, 128, 1, 2, false, c,
                         0); // N=   8192, p= 26.22 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, conjugate,
                         conjugate, 64, 64, 32, 32, 32, 16, 128, 1, 2, false, c,
                         1); // N=   2048, p= 23.91 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, conjugate,
                         conjugate, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 14.49 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, conjugate,
                         conjugate, 128, 128, 32, 64, 32, 16, 256, 1, 2, false,
                         c, 0); // N=   8192, p= 75.10 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, conjugate,
                         conjugate, 64, 64, 32, 64, 16, 32, 128, 1, 2, false, c,
                         1); // N=   2048, p= 57.81 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, conjugate,
                         conjugate, 64, 64, 64, 32, 16, 16, 256, 2, 2, false, c,
                         2); // N=    512, p= 21.11 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, conjugate,
                         conjugate, 64, 64, 32, 32, 32, 16, 128, 1, 2, false, c,
                         0); // N=   8192, p= 48.24 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, conjugate,
                         conjugate, 64, 64, 32, 32, 32, 16, 128, 1, 2, false, c,
                         1); // N=   2048, p= 46.23 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, conjugate,
                         conjugate, 32, 32, 32, 16, 16, 16, 128, 1, 2, false, c,
                         2); // N=    512, p= 19.53 [TFlop/s]
#endif

#ifdef COMPILE_SGEMM_STRIDEDBATCH_KERNEL
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      with_ec, col_major, col_major, 128, 64,
                                      32, 32, 64, 32, 128, 1, 2, false, s,
                                      0); // N=   1024, p= 44.68 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      with_ec, col_major, col_major, 128, 64,
                                      32, 32, 64, 32, 128, 1, 2, false, s,
                                      1); // N=    256, p= 24.63 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      with_ec, col_major, col_major, 64, 64, 64,
                                      32, 32, 64, 128, 1, 2, false, s,
                                      2); // N=     64, p=  6.38 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      with_ec, col_major, col_major, 128, 64,
                                      32, 64, 32, 16, 128, 1, 2, false, s,
                                      0); // N=   1024, p= 27.86 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      with_ec, col_major, col_major, 128, 64,
                                      32, 64, 32, 16, 128, 1, 2, false, s,
                                      1); // N=    256, p= 21.60 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      with_ec, col_major, col_major, 64, 64, 64,
                                      32, 32, 32, 128, 1, 2, false, s,
                                      2); // N=     64, p=  7.33 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      without_ec, col_major, col_major, 128,
                                      128, 32, 64, 64, 32, 128, 2, 2, false, s,
                                      0); // N=   1024, p= 85.03 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      without_ec, col_major, col_major, 128, 64,
                                      32, 32, 64, 32, 128, 1, 2, false, s,
                                      1); // N=    256, p= 41.12 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      without_ec, col_major, col_major, 64, 64,
                                      64, 32, 32, 64, 128, 1, 2, false, s,
                                      2); // N=     64, p=  9.22 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      without_ec, col_major, col_major, 128,
                                      128, 32, 64, 64, 16, 128, 2, 2, false, s,
                                      0); // N=   1024, p= 67.49 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      without_ec, col_major, col_major, 128, 64,
                                      32, 32, 64, 32, 128, 1, 2, false, s,
                                      1); // N=    256, p= 39.24 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      without_ec, col_major, col_major, 64, 64,
                                      64, 32, 32, 64, 128, 1, 2, false, s,
                                      2); // N=     64, p=  9.17 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      with_ec, col_major, row_major, 128, 64,
                                      32, 64, 32, 32, 128, 1, 2, false, s,
                                      0); // N=   1024, p= 38.41 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      with_ec, col_major, row_major, 128, 64,
                                      32, 64, 32, 32, 128, 1, 2, false, s,
                                      1); // N=    256, p= 26.91 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      with_ec, col_major, row_major, 64, 64, 64,
                                      32, 32, 64, 128, 1, 2, false, s,
                                      2); // N=     64, p=  8.14 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      with_ec, col_major, row_major, 128, 64,
                                      32, 32, 64, 16, 128, 2, 2, false, s,
                                      0); // N=   1024, p= 28.10 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      with_ec, col_major, row_major, 128, 64,
                                      32, 32, 64, 16, 128, 1, 2, false, s,
                                      1); // N=    256, p= 21.80 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      with_ec, col_major, row_major, 64, 64, 64,
                                      32, 32, 32, 128, 1, 2, false, s,
                                      2); // N=     64, p=  7.39 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      without_ec, col_major, row_major, 128,
                                      128, 32, 64, 64, 32, 128, 2, 2, false, s,
                                      0); // N=   1024, p= 76.12 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      without_ec, col_major, row_major, 128,
                                      128, 32, 64, 64, 16, 128, 1, 2, false, s,
                                      1); // N=    256, p= 39.12 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      without_ec, col_major, row_major, 64, 64,
                                      64, 32, 32, 64, 128, 1, 2, false, s,
                                      2); // N=     64, p=  9.02 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      without_ec, col_major, row_major, 128,
                                      128, 32, 64, 64, 32, 128, 1, 2, false, s,
                                      0); // N=   1024, p= 68.45 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      without_ec, col_major, row_major, 128, 64,
                                      32, 64, 32, 32, 128, 1, 2, false, s,
                                      1); // N=    256, p= 40.95 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      without_ec, col_major, row_major, 64, 64,
                                      64, 32, 32, 64, 128, 1, 2, false, s,
                                      2); // N=     64, p=  9.17 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      with_ec, row_major, col_major, 128, 64,
                                      32, 64, 32, 32, 128, 1, 2, false, s,
                                      0); // N=   1024, p= 47.68 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      with_ec, row_major, col_major, 128, 64,
                                      32, 64, 32, 32, 128, 1, 2, false, s,
                                      1); // N=    256, p= 33.01 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      with_ec, row_major, col_major, 64, 64, 64,
                                      32, 32, 64, 128, 1, 2, false, s,
                                      2); // N=     64, p=  8.47 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      with_ec, row_major, col_major, 128, 64,
                                      32, 32, 64, 16, 128, 1, 2, false, s,
                                      0); // N=   1024, p= 26.66 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      with_ec, row_major, col_major, 128, 64,
                                      32, 32, 64, 16, 128, 1, 2, false, s,
                                      1); // N=    256, p= 21.10 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      with_ec, row_major, col_major, 64, 64, 64,
                                      32, 32, 32, 128, 1, 2, false, s,
                                      2); // N=     64, p=  7.17 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      without_ec, row_major, col_major, 128,
                                      128, 32, 64, 64, 32, 128, 2, 2, false, s,
                                      0); // N=   1024, p= 90.90 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      without_ec, row_major, col_major, 128, 64,
                                      32, 32, 32, 32, 256, 1, 2, false, s,
                                      1); // N=    256, p= 41.68 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      without_ec, row_major, col_major, 64, 64,
                                      64, 32, 32, 64, 128, 1, 2, false, s,
                                      2); // N=     64, p=  9.64 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      without_ec, row_major, col_major, 128,
                                      128, 32, 64, 64, 32, 128, 2, 2, false, s,
                                      0); // N=   1024, p= 62.73 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      without_ec, row_major, col_major, 128, 64,
                                      32, 32, 64, 16, 128, 1, 2, false, s,
                                      1); // N=    256, p= 38.33 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      without_ec, row_major, col_major, 64, 64,
                                      64, 32, 32, 64, 128, 1, 2, false, s,
                                      2); // N=     64, p=  8.78 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      with_ec, row_major, row_major, 128, 64,
                                      32, 64, 32, 32, 128, 2, 2, false, s,
                                      0); // N=   1024, p= 45.04 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      with_ec, row_major, row_major, 128, 64,
                                      32, 64, 32, 32, 128, 1, 2, false, s,
                                      1); // N=    256, p= 30.68 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      with_ec, row_major, row_major, 64, 64, 64,
                                      32, 32, 64, 128, 1, 2, false, s,
                                      2); // N=     64, p=  8.18 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      with_ec, row_major, row_major, 128, 64,
                                      32, 32, 64, 16, 128, 1, 2, false, s,
                                      0); // N=   1024, p= 27.84 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      with_ec, row_major, row_major, 128, 64,
                                      32, 32, 64, 16, 128, 1, 2, false, s,
                                      1); // N=    256, p= 21.88 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      with_ec, row_major, row_major, 64, 64, 64,
                                      32, 32, 32, 128, 1, 2, false, s,
                                      2); // N=     64, p=  7.36 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      without_ec, row_major, row_major, 128,
                                      128, 32, 64, 64, 32, 128, 2, 2, false, s,
                                      0); // N=   1024, p= 83.69 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      without_ec, row_major, row_major, 128, 64,
                                      32, 64, 32, 32, 128, 2, 2, false, s,
                                      1); // N=    256, p= 40.49 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, half,
                                      without_ec, row_major, row_major, 64, 64,
                                      64, 32, 32, 64, 128, 1, 2, false, s,
                                      2); // N=     64, p=  9.37 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      without_ec, row_major, row_major, 128,
                                      128, 32, 64, 64, 16, 128, 1, 2, false, s,
                                      0); // N=   1024, p= 65.96 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      without_ec, row_major, row_major, 128, 64,
                                      32, 32, 64, 32, 128, 1, 2, false, s,
                                      1); // N=    256, p= 39.27 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, float, tf32,
                                      without_ec, row_major, row_major, 64, 64,
                                      64, 32, 32, 64, 128, 1, 2, false, s,
                                      2); // N=     64, p=  9.17 [TFlop/s]
#endif
#ifdef COMPILE_CGEMM_STRIDEDBATCH_KERNEL
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, col_major, col_major, 64, 64, 32,
                                      16, 64, 32, 128, 1, 2, false, c,
                                      0); // N=   1024, p= 44.93 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, col_major, col_major, 64, 64, 32,
                                      16, 64, 32, 128, 1, 2, false, c,
                                      1); // N=    256, p= 30.66 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, col_major, col_major, 64, 32, 32,
                                      16, 32, 32, 128, 2, 2, false, c,
                                      2); // N=     64, p= 14.31 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, col_major, col_major, 64, 64, 32,
                                      16, 64, 16, 128, 1, 2, false, c,
                                      0); // N=   1024, p= 30.79 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, col_major, col_major, 64, 32, 32,
                                      16, 32, 32, 128, 1, 2, false, c,
                                      1); // N=    256, p= 24.96 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, col_major, col_major, 64, 32, 32,
                                      16, 32, 16, 128, 2, 2, false, c,
                                      2); // N=     64, p= 15.43 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, col_major, col_major, 128, 64,
                                      32, 32, 64, 16, 128, 1, 2, false, c,
                                      0); // N=   1024, p= 88.76 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, col_major, col_major, 128, 64,
                                      32, 32, 64, 16, 128, 1, 2, false, c,
                                      1); // N=    256, p= 64.39 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, col_major, col_major, 64, 64,
                                      64, 16, 64, 32, 128, 1, 2, false, c,
                                      2); // N=     64, p= 24.05 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, col_major, col_major, 128, 64,
                                      32, 32, 32, 16, 256, 1, 2, false, c,
                                      0); // N=   1024, p= 61.04 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, col_major, col_major, 128, 64,
                                      32, 32, 32, 16, 256, 1, 2, false, c,
                                      1); // N=    256, p= 46.78 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, col_major, col_major, 64, 64,
                                      64, 32, 16, 32, 256, 1, 2, false, c,
                                      2); // N=     64, p= 20.34 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, col_major, row_major, 64, 64, 32,
                                      32, 32, 16, 128, 2, 2, false, c,
                                      0); // N=   1024, p= 41.29 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, col_major, row_major, 64, 64, 32,
                                      32, 32, 16, 128, 2, 2, false, c,
                                      1); // N=    256, p= 32.49 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, col_major, row_major, 64, 64, 64,
                                      32, 32, 16, 128, 1, 2, false, c,
                                      2); // N=     64, p= 17.57 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, col_major, row_major, 64, 64, 32,
                                      32, 32, 16, 128, 1, 2, false, c,
                                      0); // N=   1024, p= 28.77 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, col_major, row_major, 64, 32, 32,
                                      16, 32, 32, 128, 2, 2, false, c,
                                      1); // N=    256, p= 24.71 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, col_major, row_major, 64, 32, 32,
                                      16, 32, 16, 128, 1, 2, false, c,
                                      2); // N=     64, p= 15.32 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, col_major, row_major, 128, 64,
                                      32, 32, 64, 16, 128, 1, 2, false, c,
                                      0); // N=   1024, p= 67.23 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, col_major, row_major, 128, 64,
                                      32, 32, 64, 16, 128, 1, 2, false, c,
                                      1); // N=    256, p= 50.76 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, col_major, row_major, 64, 64,
                                      64, 32, 32, 16, 128, 1, 2, false, c,
                                      2); // N=     64, p= 21.30 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, col_major, row_major, 64, 64,
                                      32, 32, 32, 32, 128, 2, 2, false, c,
                                      0); // N=   1024, p= 57.31 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, col_major, row_major, 64, 64,
                                      32, 32, 32, 32, 128, 2, 2, false, c,
                                      1); // N=    256, p= 43.23 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, col_major, row_major, 64, 64,
                                      64, 32, 16, 32, 256, 1, 2, false, c,
                                      2); // N=     64, p= 20.34 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, col_major, conjugate, 64, 64, 32,
                                      32, 32, 16, 128, 2, 2, false, c,
                                      0); // N=   1024, p= 38.97 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, col_major, conjugate, 64, 64, 32,
                                      32, 32, 16, 128, 2, 2, false, c,
                                      1); // N=    256, p= 27.73 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, col_major, conjugate, 64, 32, 32,
                                      32, 16, 16, 128, 1, 2, false, c,
                                      2); // N=     64, p= 15.50 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, col_major, conjugate, 64, 32, 32,
                                      16, 32, 32, 128, 1, 2, false, c,
                                      0); // N=   1024, p= 27.11 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, col_major, conjugate, 64, 32, 32,
                                      16, 32, 32, 128, 1, 2, false, c,
                                      1); // N=    256, p= 22.45 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, col_major, conjugate, 64, 64, 64,
                                      16, 32, 16, 256, 2, 2, false, c,
                                      2); // N=     64, p= 14.28 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, col_major, conjugate, 128,
                                      128, 32, 32, 64, 16, 256, 1, 2, false, c,
                                      0); // N=   1024, p= 59.50 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, col_major, conjugate, 128,
                                      128, 32, 32, 64, 16, 256, 1, 2, false, c,
                                      1); // N=    256, p= 36.40 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, col_major, conjugate, 64, 64,
                                      64, 32, 32, 16, 128, 1, 2, false, c,
                                      2); // N=     64, p= 17.21 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, col_major, conjugate, 64, 64,
                                      32, 32, 32, 32, 128, 1, 2, false, c,
                                      0); // N=   1024, p= 52.08 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, col_major, conjugate, 64, 64,
                                      32, 32, 16, 16, 256, 2, 2, false, c,
                                      1); // N=    256, p= 37.48 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, col_major, conjugate, 64, 64,
                                      64, 32, 16, 32, 256, 2, 2, false, c,
                                      2); // N=     64, p= 18.75 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, row_major, col_major, 64, 64, 32,
                                      32, 32, 16, 128, 2, 2, false, c,
                                      0); // N=   1024, p= 51.34 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, row_major, col_major, 64, 64, 32,
                                      32, 32, 16, 128, 1, 2, false, c,
                                      1); // N=    256, p= 39.90 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, row_major, col_major, 64, 32, 32,
                                      32, 16, 32, 128, 2, 2, false, c,
                                      2); // N=     64, p= 19.74 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, row_major, col_major, 64, 64, 32,
                                      16, 64, 16, 128, 1, 2, false, c,
                                      0); // N=   1024, p= 31.33 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, row_major, col_major, 64, 64, 32,
                                      16, 64, 16, 128, 1, 2, false, c,
                                      1); // N=    256, p= 25.08 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, row_major, col_major, 64, 64, 64,
                                      16, 32, 16, 256, 2, 2, false, c,
                                      2); // N=     64, p= 15.43 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, row_major, col_major, 128, 64,
                                      32, 64, 32, 16, 128, 1, 2, false, c,
                                      0); // N=   1024, p=119.62 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, row_major, col_major, 128, 64,
                                      32, 32, 64, 16, 128, 1, 2, false, c,
                                      1); // N=    256, p= 77.28 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, row_major, col_major, 64, 64,
                                      64, 64, 16, 32, 128, 2, 2, false, c,
                                      2); // N=     64, p= 25.91 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, row_major, col_major, 128, 64,
                                      32, 32, 32, 16, 256, 1, 2, false, c,
                                      0); // N=   1024, p= 59.02 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, row_major, col_major, 128, 64,
                                      32, 32, 32, 16, 256, 1, 2, false, c,
                                      1); // N=    256, p= 45.77 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, row_major, col_major, 64, 32,
                                      32, 32, 16, 16, 128, 1, 2, false, c,
                                      2); // N=     64, p= 20.46 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, row_major, row_major, 64, 64, 32,
                                      32, 32, 16, 128, 2, 2, false, c,
                                      0); // N=   1024, p= 47.92 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, row_major, row_major, 64, 64, 32,
                                      32, 32, 16, 128, 1, 2, false, c,
                                      1); // N=    256, p= 38.19 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, row_major, row_major, 64, 64, 64,
                                      32, 32, 16, 128, 2, 2, false, c,
                                      2); // N=     64, p= 19.17 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, row_major, row_major, 64, 64, 32,
                                      16, 64, 16, 128, 1, 2, false, c,
                                      0); // N=   1024, p= 31.18 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, row_major, row_major, 64, 64, 32,
                                      16, 64, 16, 128, 1, 2, false, c,
                                      1); // N=    256, p= 24.90 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, row_major, row_major, 64, 32, 32,
                                      16, 32, 16, 128, 2, 2, false, c,
                                      2); // N=     64, p= 15.43 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, row_major, row_major, 128, 64,
                                      32, 64, 32, 16, 128, 2, 2, false, c,
                                      0); // N=   1024, p= 90.58 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, row_major, row_major, 128, 64,
                                      32, 64, 32, 32, 128, 1, 2, false, c,
                                      1); // N=    256, p= 65.40 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, row_major, row_major, 64, 64,
                                      64, 64, 16, 32, 128, 2, 2, false, c,
                                      2); // N=     64, p= 24.67 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, row_major, row_major, 128, 64,
                                      32, 32, 32, 16, 256, 1, 2, false, c,
                                      0); // N=   1024, p= 58.91 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, row_major, row_major, 128, 64,
                                      32, 32, 32, 16, 256, 1, 2, false, c,
                                      1); // N=    256, p= 45.61 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, row_major, row_major, 64, 32,
                                      32, 32, 16, 16, 128, 1, 2, false, c,
                                      2); // N=     64, p= 20.46 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, row_major, conjugate, 64, 64, 32,
                                      32, 32, 16, 128, 1, 2, false, c,
                                      0); // N=   1024, p= 44.86 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, row_major, conjugate, 128, 64,
                                      32, 32, 32, 16, 256, 1, 2, false, c,
                                      1); // N=    256, p= 31.60 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, row_major, conjugate, 64, 32, 32,
                                      32, 16, 32, 128, 1, 2, false, c,
                                      2); // N=     64, p= 17.03 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, row_major, conjugate, 64, 64, 32,
                                      32, 32, 16, 128, 2, 2, false, c,
                                      0); // N=   1024, p= 27.46 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, row_major, conjugate, 64, 32, 32,
                                      16, 32, 32, 128, 1, 2, false, c,
                                      1); // N=    256, p= 22.66 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, row_major, conjugate, 64, 64, 64,
                                      16, 32, 16, 256, 2, 2, false, c,
                                      2); // N=     64, p= 13.89 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, row_major, conjugate, 128,
                                      128, 32, 64, 32, 16, 256, 1, 2, false, c,
                                      0); // N=   1024, p= 76.47 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, row_major, conjugate, 128, 64,
                                      32, 64, 16, 16, 256, 1, 2, false, c,
                                      1); // N=    256, p= 46.43 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, row_major, conjugate, 64, 64,
                                      64, 32, 16, 16, 256, 2, 2, false, c,
                                      2); // N=     64, p= 20.84 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, row_major, conjugate, 64, 64,
                                      32, 32, 32, 32, 128, 1, 2, false, c,
                                      0); // N=   1024, p= 52.63 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, row_major, conjugate, 128, 64,
                                      32, 32, 32, 16, 256, 1, 2, false, c,
                                      1); // N=    256, p= 36.62 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, row_major, conjugate, 64, 64,
                                      64, 32, 16, 64, 256, 1, 2, false, c,
                                      2); // N=     64, p= 18.44 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, conjugate, col_major, 64, 64, 32,
                                      32, 32, 16, 128, 1, 2, false, c,
                                      0); // N=   1024, p= 47.84 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, conjugate, col_major, 64, 64, 32,
                                      32, 32, 16, 128, 1, 2, false, c,
                                      1); // N=    256, p= 32.18 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, conjugate, col_major, 64, 64, 64,
                                      32, 16, 16, 256, 2, 2, false, c,
                                      2); // N=     64, p= 17.61 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, conjugate, col_major, 64, 64, 32,
                                      16, 64, 16, 128, 1, 2, false, c,
                                      0); // N=   1024, p= 29.00 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, conjugate, col_major, 64, 64, 32,
                                      16, 64, 16, 128, 1, 2, false, c,
                                      1); // N=    256, p= 21.77 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, conjugate, col_major, 64, 64, 64,
                                      16, 32, 16, 256, 2, 2, false, c,
                                      2); // N=     64, p= 14.19 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, conjugate, col_major, 128,
                                      128, 32, 64, 32, 16, 256, 2, 2, false, c,
                                      0); // N=   1024, p= 95.88 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, conjugate, col_major, 128,
                                      128, 32, 64, 32, 16, 256, 1, 2, false, c,
                                      1); // N=    256, p= 47.06 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, conjugate, col_major, 64, 64,
                                      64, 32, 16, 16, 256, 2, 2, false, c,
                                      2); // N=     64, p= 22.98 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, conjugate, col_major, 64, 64,
                                      32, 32, 32, 16, 128, 1, 2, false, c,
                                      0); // N=   1024, p= 53.66 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, conjugate, col_major, 64, 64,
                                      32, 32, 32, 16, 128, 1, 2, false, c,
                                      1); // N=    256, p= 34.70 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, conjugate, col_major, 64, 64,
                                      64, 32, 16, 64, 256, 1, 2, false, c,
                                      2); // N=     64, p= 18.34 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, conjugate, row_major, 64, 64, 32,
                                      32, 32, 16, 128, 2, 2, false, c,
                                      0); // N=   1024, p= 45.18 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, conjugate, row_major, 64, 64, 32,
                                      32, 32, 16, 128, 2, 2, false, c,
                                      1); // N=    256, p= 30.69 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, conjugate, row_major, 64, 64, 64,
                                      16, 32, 16, 256, 2, 2, false, c,
                                      2); // N=     64, p= 17.16 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, conjugate, row_major, 64, 64, 32,
                                      32, 32, 16, 128, 2, 2, false, c,
                                      0); // N=   1024, p= 27.39 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, conjugate, row_major, 32, 32, 32,
                                      16, 16, 16, 128, 1, 2, false, c,
                                      1); // N=    256, p= 20.94 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, conjugate, row_major, 64, 64, 64,
                                      16, 32, 16, 256, 1, 2, false, c,
                                      2); // N=     64, p= 14.28 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, conjugate, row_major, 128,
                                      128, 32, 64, 32, 16, 256, 1, 2, false, c,
                                      0); // N=   1024, p= 78.89 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, conjugate, row_major, 128,
                                      128, 32, 64, 32, 16, 256, 1, 2, false, c,
                                      1); // N=    256, p= 41.69 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, conjugate, row_major, 64, 64,
                                      64, 32, 16, 16, 256, 2, 2, false, c,
                                      2); // N=     64, p= 20.84 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, conjugate, row_major, 64, 64,
                                      32, 32, 32, 32, 128, 1, 2, false, c,
                                      0); // N=   1024, p= 53.13 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, conjugate, row_major, 64, 64,
                                      32, 32, 32, 32, 128, 1, 2, false, c,
                                      1); // N=    256, p= 34.32 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, conjugate, row_major, 64, 64,
                                      64, 32, 16, 64, 256, 2, 2, false, c,
                                      2); // N=     64, p= 18.49 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, conjugate, conjugate, 64, 64, 32,
                                      32, 32, 16, 128, 1, 2, false, c,
                                      0); // N=   1024, p= 41.35 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, conjugate, conjugate, 64, 64, 32,
                                      32, 32, 16, 128, 1, 2, false, c,
                                      1); // N=    256, p= 26.40 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      with_ec, conjugate, conjugate, 64, 64, 64,
                                      32, 16, 32, 256, 2, 2, false, c,
                                      2); // N=     64, p= 15.90 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, conjugate, conjugate, 64, 64, 32,
                                      32, 32, 16, 128, 1, 2, false, c,
                                      0); // N=   1024, p= 25.68 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, conjugate, conjugate, 32, 32, 32,
                                      16, 16, 16, 128, 1, 2, false, c,
                                      1); // N=    256, p= 18.68 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      with_ec, conjugate, conjugate, 64, 64, 64,
                                      16, 32, 16, 256, 1, 2, false, c,
                                      2); // N=     64, p= 13.29 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, conjugate, conjugate, 128,
                                      128, 32, 64, 32, 16, 256, 1, 2, false, c,
                                      0); // N=   1024, p= 70.08 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, conjugate, conjugate, 128,
                                      128, 32, 64, 32, 16, 256, 1, 2, false, c,
                                      1); // N=    256, p= 37.41 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, half,
                                      without_ec, conjugate, conjugate, 64, 64,
                                      64, 32, 16, 16, 256, 2, 2, false, c,
                                      2); // N=     64, p= 19.07 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, conjugate, conjugate, 64, 64,
                                      32, 32, 32, 16, 128, 2, 2, false, c,
                                      0); // N=   1024, p= 47.19 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, conjugate, conjugate, 32, 32,
                                      32, 16, 16, 16, 128, 1, 2, false, c,
                                      1); // N=    256, p= 28.63 [TFlop/s]
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(gemm_stridedBatch_module, cuComplex, tf32,
                                      without_ec, conjugate, conjugate, 64, 32,
                                      32, 16, 16, 16, 256, 1, 2, false, c,
                                      2); // N=     64, p= 17.48 [TFlop/s]
#endif
#ifdef COMPILE_SGEMM_ATOMIC_KERNEL
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, float, half, with_ec, col_major, col_major, 64, 64,
      32, 64, 32, 32, 32, 128, 1, 2, false,
      s); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, float, tf32, with_ec, col_major, col_major, 64, 64,
      32, 64, 32, 32, 32, 128, 1, 2, false,
      s); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, float, half, without_ec, col_major, col_major, 64, 64,
      32, 64, 32, 32, 32, 128, 1, 2, false,
      s); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, float, tf32, without_ec, col_major, col_major, 64, 64,
      32, 64, 32, 32, 32, 128, 1, 2, false,
      s); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, float, half, with_ec, col_major, row_major, 64, 64,
      32, 64, 32, 32, 32, 128, 1, 2, false,
      s); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, float, tf32, with_ec, col_major, row_major, 64, 64,
      32, 64, 32, 32, 32, 128, 1, 2, false,
      s); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, float, half, without_ec, col_major, row_major, 64, 64,
      32, 64, 32, 32, 32, 128, 1, 2, false,
      s); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, float, tf32, without_ec, col_major, row_major, 64, 64,
      32, 64, 32, 32, 32, 128, 1, 2, false,
      s); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, float, half, with_ec, row_major, col_major, 64, 64,
      32, 64, 32, 32, 32, 128, 1, 2, false,
      s); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, float, tf32, with_ec, row_major, col_major, 64, 64,
      32, 64, 32, 32, 32, 128, 1, 2, false,
      s); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, float, half, without_ec, row_major, col_major, 64, 64,
      32, 64, 32, 32, 32, 128, 1, 2, false,
      s); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, float, tf32, without_ec, row_major, col_major, 64, 64,
      32, 64, 32, 32, 32, 128, 1, 2, false,
      s); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, float, half, with_ec, row_major, row_major, 64, 64,
      32, 64, 32, 32, 32, 128, 1, 2, false,
      s); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, float, tf32, with_ec, row_major, row_major, 64, 64,
      32, 64, 32, 32, 32, 128, 1, 2, false,
      s); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, float, half, without_ec, row_major, row_major, 64, 64,
      32, 64, 32, 32, 32, 128, 1, 2, false,
      s); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, float, tf32, without_ec, row_major, row_major, 64, 64,
      32, 64, 32, 32, 32, 128, 1, 2, false,
      s); // Not optimized but works on any Ampere GPUs
#endif
#ifdef COMPILE_CGEMM_ATOMIC_KERNEL
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, half, with_ec, col_major, col_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, tf32, with_ec, col_major, col_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, half, without_ec, col_major, col_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, tf32, without_ec, col_major, col_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, half, with_ec, col_major, row_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, tf32, with_ec, col_major, row_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, half, without_ec, col_major, row_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, tf32, without_ec, col_major, row_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, half, with_ec, col_major, conjugate, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, tf32, with_ec, col_major, conjugate, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, half, without_ec, col_major, conjugate, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, tf32, without_ec, col_major, conjugate, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, half, with_ec, row_major, col_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, tf32, with_ec, row_major, col_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, half, without_ec, row_major, col_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, tf32, without_ec, row_major, col_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, half, with_ec, row_major, row_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, tf32, with_ec, row_major, row_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, half, without_ec, row_major, row_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, tf32, without_ec, row_major, row_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, half, with_ec, row_major, conjugate, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, tf32, with_ec, row_major, conjugate, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, half, without_ec, row_major, conjugate, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, tf32, without_ec, row_major, conjugate, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, half, with_ec, conjugate, col_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, tf32, with_ec, conjugate, col_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, half, without_ec, conjugate, col_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, tf32, without_ec, conjugate, col_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, half, with_ec, conjugate, row_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, tf32, with_ec, conjugate, row_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, half, without_ec, conjugate, row_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, tf32, without_ec, conjugate, row_major, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, half, with_ec, conjugate, conjugate, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, tf32, with_ec, conjugate, conjugate, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, half, without_ec, conjugate, conjugate, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
  SET_GEMM_ATOMIC_KERNEL_MODULE(
      gemm_atomic_module, cuComplex, tf32, without_ec, conjugate, conjugate, 64,
      64, 32, 64, 32, 32, 32, 128, 1, 2, false,
      c); // Not optimized but works on any Ampere GPUs
#endif
}
