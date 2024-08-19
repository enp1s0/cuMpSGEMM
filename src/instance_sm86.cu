#include "cumpsgemm_kernel.cuh"
#include "handle.hpp"
#include "instance.hpp"

void cumpsgemm::configure_instance_sm86(
    cumpsgemm::gemm_module gemm_module[cumpsgemm::kernel_module_code::max_code]
                                      [cumpsgemm::num_kernel_candidates],
    cumpsgemm::gemm_module
        gemm_stridedBatch_module[cumpsgemm::kernel_module_code::max_code]
                                [cumpsgemm::num_kernel_candidates],
    cumpsgemm::gemm_module
        gemm_atomic_module[cumpsgemm::kernel_module_code::max_code]) {
  using tf32 = nvcuda::wmma::precision::tf32;

  // Optimized ion A6000
#ifdef COMPILE_SGEMM_KERNEL
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, col_major,
                         col_major, 128, 128, 32, 32, 64, 32, 256, 1, 2, false,
                         s, 0); // N=  16384, p= 25.55 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, col_major,
                         col_major, 128, 128, 32, 32, 64, 32, 256, 1, 2, false,
                         s, 1); // N=   4096, p= 29.55 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, col_major,
                         col_major, 128, 128, 32, 32, 64, 32, 256, 1, 2, false,
                         s, 2); // N=   1024, p= 21.26 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, col_major,
                         col_major, 64, 128, 32, 64, 32, 16, 128, 2, 2, false,
                         s, 0); // N=  16384, p= 16.40 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, col_major,
                         col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
                         1); // N=   4096, p= 19.38 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, col_major,
                         col_major, 128, 32, 32, 32, 32, 16, 128, 1, 2, false,
                         s, 2); // N=   1024, p= 14.46 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, col_major,
                         col_major, 128, 128, 32, 32, 64, 16, 256, 2, 2, false,
                         s, 0); // N=  16384, p= 41.76 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, col_major,
                         col_major, 128, 128, 32, 32, 64, 16, 256, 2, 2, false,
                         s, 1); // N=   4096, p= 51.35 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, col_major,
                         col_major, 128, 128, 32, 64, 64, 32, 128, 1, 2, false,
                         s, 2); // N=   1024, p= 37.68 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, col_major,
                         col_major, 128, 128, 32, 64, 32, 32, 256, 2, 2, false,
                         s, 0); // N=  16384, p= 37.01 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, col_major,
                         col_major, 128, 128, 32, 64, 32, 16, 256, 2, 2, false,
                         s, 1); // N=   4096, p= 43.53 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, col_major,
                         col_major, 128, 128, 32, 64, 64, 32, 128, 2, 2, false,
                         s, 2); // N=   1024, p= 30.25 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, col_major,
                         row_major, 64, 128, 32, 32, 32, 32, 256, 1, 2, false,
                         s, 0); // N=  16384, p= 22.82 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, col_major,
                         row_major, 64, 128, 32, 32, 32, 32, 256, 1, 2, false,
                         s, 1); // N=   4096, p= 26.15 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, col_major,
                         row_major, 32, 128, 32, 32, 32, 16, 128, 1, 2, false,
                         s, 2); // N=   1024, p= 18.92 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, col_major,
                         row_major, 128, 128, 32, 64, 32, 16, 256, 2, 2, false,
                         s, 0); // N=  16384, p= 16.71 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, col_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, s,
                         1); // N=   4096, p= 19.62 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, col_major,
                         row_major, 128, 64, 32, 32, 64, 16, 128, 1, 2, false,
                         s, 2); // N=   1024, p= 13.55 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, col_major,
                         row_major, 128, 128, 32, 32, 64, 32, 256, 2, 2, false,
                         s, 0); // N=  16384, p= 40.57 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, col_major,
                         row_major, 128, 128, 32, 64, 64, 32, 128, 1, 2, false,
                         s, 1); // N=   4096, p= 47.92 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, col_major,
                         row_major, 128, 128, 32, 64, 64, 16, 128, 1, 2, false,
                         s, 2); // N=   1024, p= 33.29 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, col_major,
                         row_major, 128, 128, 32, 64, 32, 32, 256, 2, 2, false,
                         s, 0); // N=  16384, p= 36.95 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, col_major,
                         row_major, 128, 128, 32, 64, 32, 32, 256, 1, 2, false,
                         s, 1); // N=   4096, p= 45.80 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, col_major,
                         row_major, 128, 128, 32, 64, 32, 32, 256, 2, 2, false,
                         s, 2); // N=   1024, p= 28.07 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, row_major,
                         col_major, 128, 128, 32, 64, 32, 32, 256, 1, 2, false,
                         s, 0); // N=  16384, p= 25.34 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, row_major,
                         col_major, 128, 64, 32, 64, 32, 32, 128, 2, 2, false,
                         s, 1); // N=   4096, p= 29.39 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, row_major,
                         col_major, 128, 128, 32, 64, 32, 32, 256, 1, 2, false,
                         s, 2); // N=   1024, p= 19.43 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, row_major,
                         col_major, 128, 128, 32, 64, 32, 16, 256, 2, 2, false,
                         s, 0); // N=  16384, p= 16.02 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, row_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, s,
                         1); // N=   4096, p= 19.06 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, row_major,
                         col_major, 128, 128, 32, 64, 32, 16, 256, 2, 2, false,
                         s, 2); // N=   1024, p= 12.06 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, row_major,
                         col_major, 128, 128, 32, 64, 32, 32, 256, 1, 2, false,
                         s, 0); // N=  16384, p= 38.10 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, row_major,
                         col_major, 128, 128, 32, 64, 64, 32, 128, 1, 2, false,
                         s, 1); // N=   4096, p= 54.76 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, row_major,
                         col_major, 128, 128, 32, 32, 64, 32, 256, 1, 2, false,
                         s, 2); // N=   1024, p= 36.40 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, row_major,
                         col_major, 128, 128, 32, 64, 32, 32, 256, 1, 2, false,
                         s, 0); // N=  16384, p= 36.75 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, row_major,
                         col_major, 128, 128, 32, 64, 32, 32, 256, 1, 2, false,
                         s, 1); // N=   4096, p= 43.40 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, row_major,
                         col_major, 128, 128, 32, 64, 32, 32, 256, 1, 2, false,
                         s, 2); // N=   1024, p= 28.26 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, row_major,
                         row_major, 64, 128, 32, 64, 32, 32, 128, 2, 2, false,
                         s, 0); // N=  16384, p= 24.85 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, row_major,
                         row_major, 128, 64, 32, 64, 32, 32, 128, 2, 2, false,
                         s, 1); // N=   4096, p= 28.94 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, with_ec, row_major,
                         row_major, 128, 64, 32, 64, 32, 32, 128, 2, 2, false,
                         s, 2); // N=   1024, p= 19.00 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, row_major,
                         row_major, 128, 128, 32, 64, 32, 16, 256, 2, 2, false,
                         s, 0); // N=  16384, p= 16.26 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, row_major,
                         row_major, 32, 128, 32, 32, 32, 32, 128, 2, 2, false,
                         s, 1); // N=   4096, p= 18.80 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, with_ec, row_major,
                         row_major, 64, 128, 32, 32, 32, 32, 256, 2, 2, false,
                         s, 2); // N=   1024, p= 12.74 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, row_major,
                         row_major, 64, 128, 32, 64, 16, 32, 256, 2, 2, false,
                         s, 0); // N=  16384, p= 35.47 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, row_major,
                         row_major, 128, 128, 32, 64, 32, 16, 256, 2, 2, false,
                         s, 1); // N=   4096, p= 50.54 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, half, without_ec, row_major,
                         row_major, 128, 128, 32, 64, 64, 32, 128, 1, 2, false,
                         s, 2); // N=   1024, p= 33.55 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, row_major,
                         row_major, 128, 128, 32, 64, 32, 32, 256, 1, 2, false,
                         s, 0); // N=  16384, p= 35.48 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, row_major,
                         row_major, 128, 128, 32, 64, 32, 32, 256, 1, 2, false,
                         s, 1); // N=   4096, p= 42.76 [TFlop/s]
  SET_GEMM_KERNEL_MODULE(gemm_module, float, tf32, without_ec, row_major,
                         row_major, 128, 128, 32, 64, 32, 32, 256, 1, 2, false,
                         s, 2); // N=   1024, p= 28.44 [TFlop/s]
#endif

#ifdef COMPILE_CGEMM_KERNEL
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, col_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, col_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, col_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, col_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, col_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, col_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, col_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, col_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, col_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, col_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, col_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, col_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, col_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, col_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, col_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, col_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, col_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, col_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, col_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, col_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, col_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, col_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, col_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, col_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, col_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, col_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, col_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, col_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, col_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, col_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, col_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, col_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, col_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, col_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, col_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, col_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, row_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, row_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, row_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, row_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, row_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, row_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, row_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, row_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, row_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, row_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, row_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, row_major,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, row_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, row_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, row_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, row_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, row_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, row_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, row_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, row_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, row_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, row_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, row_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, row_major,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, row_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, row_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, row_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, row_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, row_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, row_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, row_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, row_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, row_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, row_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, row_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, row_major,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, conjugate,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, conjugate,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, conjugate,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, conjugate,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, conjugate,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, conjugate,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, conjugate,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, conjugate,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, conjugate,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, conjugate,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, conjugate,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, conjugate,
                         col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, conjugate,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, conjugate,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, conjugate,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, conjugate,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, conjugate,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, conjugate,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, conjugate,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, conjugate,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, conjugate,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, conjugate,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, conjugate,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, conjugate,
                         row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, conjugate,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, conjugate,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, with_ec, conjugate,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, conjugate,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, conjugate,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, with_ec, conjugate,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, conjugate,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, conjugate,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, half, without_ec, conjugate,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, conjugate,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, conjugate,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_KERNEL_MODULE(gemm_module, cuComplex, tf32, without_ec, conjugate,
                         conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c,
                         2); // Not optimized but works on any Ampere GPUs
#endif

#ifdef COMPILE_SGEMM_STRIDEDBATCH_KERNEL
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, with_ec, col_major, col_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, with_ec, col_major, col_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, with_ec, col_major, col_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, with_ec, col_major, col_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, with_ec, col_major, col_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, with_ec, col_major, col_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, without_ec, col_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, without_ec, col_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, without_ec, col_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, without_ec, col_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, without_ec, col_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, without_ec, col_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, with_ec, col_major, row_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, with_ec, col_major, row_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, with_ec, col_major, row_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, with_ec, col_major, row_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, with_ec, col_major, row_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, with_ec, col_major, row_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, without_ec, col_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, without_ec, col_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, without_ec, col_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, without_ec, col_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, without_ec, col_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, without_ec, col_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, with_ec, row_major, col_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, with_ec, row_major, col_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, with_ec, row_major, col_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, with_ec, row_major, col_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, with_ec, row_major, col_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, with_ec, row_major, col_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, without_ec, row_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, without_ec, row_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, without_ec, row_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, without_ec, row_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, without_ec, row_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, without_ec, row_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, with_ec, row_major, row_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, with_ec, row_major, row_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, with_ec, row_major, row_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, with_ec, row_major, row_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, with_ec, row_major, row_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, with_ec, row_major, row_major, 64,
      64, 32, 32, 32, 16, 128, 2, 2, false, s,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, without_ec, row_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, without_ec, row_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, half, without_ec, row_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, without_ec, row_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, without_ec, row_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, float, tf32, without_ec, row_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, s,
      2); // Not optimized but works on any Ampere GPUs
#endif
#ifdef COMPILE_CGEMM_STRIDEDBATCH_KERNEL
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, col_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, col_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, col_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, col_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, col_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, col_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, col_major,
      col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, col_major,
      col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, col_major,
      col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major,
      col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major,
      col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major,
      col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, col_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, col_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, col_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, col_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, col_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, col_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, col_major,
      row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, col_major,
      row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, col_major,
      row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major,
      row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major,
      row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major,
      row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, col_major, conjugate,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, col_major, conjugate,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, col_major, conjugate,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, col_major, conjugate,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, col_major, conjugate,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, col_major, conjugate,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, col_major,
      conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, col_major,
      conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, col_major,
      conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major,
      conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major,
      conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major,
      conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, row_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, row_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, row_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, row_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, row_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, row_major, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, row_major,
      col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, row_major,
      col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, row_major,
      col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major,
      col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major,
      col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major,
      col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, row_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, row_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, row_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, row_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, row_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, row_major, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, row_major,
      row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, row_major,
      row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, row_major,
      row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major,
      row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major,
      row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major,
      row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, row_major, conjugate,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, row_major, conjugate,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, row_major, conjugate,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, row_major, conjugate,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, row_major, conjugate,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, row_major, conjugate,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, row_major,
      conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, row_major,
      conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, row_major,
      conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major,
      conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major,
      conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major,
      conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, conjugate, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, conjugate, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, conjugate, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, conjugate, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, conjugate, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, conjugate, col_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate,
      col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate,
      col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate,
      col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate,
      col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate,
      col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate,
      col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, conjugate, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, conjugate, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, conjugate, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, conjugate, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, conjugate, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, conjugate, row_major,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate,
      row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate,
      row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate,
      row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate,
      row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate,
      row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate,
      row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, conjugate, conjugate,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, conjugate, conjugate,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, with_ec, conjugate, conjugate,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, conjugate, conjugate,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, conjugate, conjugate,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, with_ec, conjugate, conjugate,
      64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate,
      conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate,
      conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate,
      conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate,
      conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      0); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate,
      conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      1); // Not optimized but works on any Ampere GPUs
  SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(
      gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate,
      conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c,
      2); // Not optimized but works on any Ampere GPUs
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
