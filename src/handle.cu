#include <cstddef>
#include <cumpsgemm/cumpsgemm.h>
#include "handle.hpp"
#include "cumpsgemm_kernel.cuh"

#define SET_GEMM_KERNEL_MODULE(module_list, io_t, tc_t, ec, op_a, op_b, smem_m, smem_n, smem_k, frag_m, frag_n, frag_k, block_size, num_unrollings, num_stages, pipelined, gemm_type, stage) \
	module_list[cumpsgemm::kernel_module_code::tc_t | cumpsgemm::kernel_module_code::ec | cumpsgemm::kernel_module_code::op_a_##op_a | cumpsgemm::kernel_module_code::op_b_##op_b | cumpsgemm::kernel_module_code::gemm_type][stage] =\
	cumpsgemm::generate_gemm_module<io_t,smem_m,smem_n,smem_k,frag_m,frag_n,frag_k,block_size,num_unrollings,num_stages,cumpsgemm::op_a,cumpsgemm::op_b,tc_t,mtk::wmma::tcec::ec, pipelined>();

#define SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(module_list, io_t, tc_t, ec, op_a, op_b, smem_m, smem_n, smem_k, frag_m, frag_n, frag_k, block_size, num_unrollings, num_stages, pipelined, gemm_type, stage) \
	module_list[cumpsgemm::kernel_module_code::tc_t | cumpsgemm::kernel_module_code::ec | cumpsgemm::kernel_module_code::op_a_##op_a | cumpsgemm::kernel_module_code::op_b_##op_b | cumpsgemm::kernel_module_code::gemm_type][stage] =\
	cumpsgemm::generate_gemm_stridedBatch_module<io_t,smem_m,smem_n,smem_k,frag_m,frag_n,frag_k,block_size,num_unrollings,num_stages,cumpsgemm::op_a,cumpsgemm::op_b,tc_t,mtk::wmma::tcec::ec, pipelined>();

#define COMPILE_SGEMM_KERNEL
#define COMPILE_CGEMM_KERNEL
#define COMPILE_SGEMM_STRIDEDBATCH_KERNEL
#define COMPILE_CGEMM_STRIDEDBATCH_KERNEL

#define ENABLE_A100_OPTIMAZED_PARAMETERS

extern "C" {
cublasStatus_t cuMpSGEMM_create(cuMpSGEMM_handle_t* const handle) {
	if ((*handle = new cuMpSGEMM_handle) == nullptr) {
		return CUBLAS_STATUS_INTERNAL_ERROR;
	}

	int num_sms;
	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
	(*handle)->num_sms = num_sms;

	using tf32 = nvcuda::wmma::precision::tf32;

#ifdef ENABLE_A100_OPTIMAZED_PARAMETERS
	// set kernel modules
#ifdef COMPILE_SGEMM_KERNEL
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , col_major, col_major,  64, 128,  32,  32,  64,  32, 128,   2,   2, false, s, 0); // N=  16384, p= 46.50 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , col_major, col_major, 128,  64,  32,  32,  64,  32, 128,   2,   2, false, s, 1); // N=   4096, p= 42.38 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , col_major, col_major,  64,  64,  32,  32,  32,  32, 128,   1,   2, false, s, 2); // N=   1024, p= 21.66 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , col_major, col_major,  64, 128,  32,  32,  64,  16, 128,   1,   2, false, s, 0); // N=  16384, p= 29.85 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , col_major, col_major,  64, 128,  32,  32,  64,  16, 128,   1,   2, false, s, 1); // N=   4096, p= 29.70 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , col_major, col_major,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, s, 2); // N=   1024, p= 19.80 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, col_major, col_major, 128, 128,  32,  64,  64,  16, 128,   1,   2, false, s, 0); // N=  16384, p= 75.47 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, col_major, col_major, 128, 128,  32,  64,  64,  32, 128,   1,   2, false, s, 1); // N=   4096, p= 88.33 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, col_major, col_major, 128, 128,  32,  64,  64,  32, 128,   1,   2, false, s, 2); // N=   1024, p= 43.91 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, col_major, col_major,  64, 128,  32,  32,  64,  16, 128,   1,   2, false, s, 0); // N=  16384, p= 64.92 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, col_major, col_major, 128, 128,  32,  64,  64,  32, 128,   2,   2, false, s, 1); // N=   4096, p= 69.45 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, col_major, col_major, 128, 128,  32,  64,  64,  32, 128,   1,   2, false, s, 2); // N=   1024, p= 37.51 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , col_major, row_major, 128,  64,  32,  64,  32,  32, 128,   1,   2, false, s, 0); // N=  16384, p= 39.40 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , col_major, row_major, 128,  64,  32,  64,  32,  32, 128,   1,   2, false, s, 1); // N=   4096, p= 40.31 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , col_major, row_major, 128,  32,  32,  32,  32,  16, 128,   2,   2, false, s, 2); // N=   1024, p= 24.61 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , col_major, row_major,  64, 128,  32,  32,  64,  16, 128,   1,   2, false, s, 0); // N=  16384, p= 30.48 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , col_major, row_major,  64, 128,  32,  32,  64,  16, 128,   1,   2, false, s, 1); // N=   4096, p= 30.52 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , col_major, row_major,  64,  64,  32,  32,  32,  32, 128,   1,   2, false, s, 2); // N=   1024, p= 18.63 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, col_major, row_major, 128, 128,  32,  64,  64,  16, 128,   1,   2, false, s, 0); // N=  16384, p= 68.42 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, col_major, row_major, 128, 128,  32,  64,  64,  32, 128,   1,   2, false, s, 1); // N=   4096, p= 76.96 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, col_major, row_major, 128, 128,  32,  64,  64,  32, 128,   1,   2, false, s, 2); // N=   1024, p= 39.29 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, col_major, row_major, 128, 128,  32,  64,  64,  32, 128,   2,   2, false, s, 0); // N=  16384, p= 65.25 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, col_major, row_major, 128, 128,  32,  64,  64,  32, 128,   2,   2, false, s, 1); // N=   4096, p= 70.62 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, col_major, row_major, 128, 128,  32,  64,  64,  32, 128,   1,   2, false, s, 2); // N=   1024, p= 37.84 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , row_major, col_major,  64, 128,  32,  32,  64,  32, 128,   1,   2, false, s, 0); // N=  16384, p= 48.65 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , row_major, col_major,  64, 128,  32,  64,  32,  32, 128,   1,   2, false, s, 1); // N=   4096, p= 50.54 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , row_major, col_major, 128,  32,  32,  32,  32,  32, 128,   1,   2, false, s, 2); // N=   1024, p= 27.59 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , row_major, col_major,  64, 128,  32,  64,  32,  16, 128,   1,   2, false, s, 0); // N=  16384, p= 29.85 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , row_major, col_major,  64, 128,  32,  64,  32,  16, 128,   1,   2, false, s, 1); // N=   4096, p= 29.90 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , row_major, col_major,  32,  64,  32,  32,  16,  16, 128,   2,   2, false, s, 2); // N=   1024, p= 18.07 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, row_major, col_major,  64, 128,  32,  32,  64,  16, 128,   1,   2, false, s, 0); // N=  16384, p= 73.16 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, row_major, col_major, 128, 128,  32,  64,  64,  32, 128,   1,   2, false, s, 1); // N=   4096, p= 96.84 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, row_major, col_major, 128, 128,  32,  64,  64,  32, 128,   1,   2, false, s, 2); // N=   1024, p= 49.44 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, row_major, col_major, 128, 128,  32,  64,  64,  32, 128,   2,   2, false, s, 0); // N=  16384, p= 60.76 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, row_major, col_major, 128, 128,  32,  64,  64,  32, 128,   2,   2, false, s, 1); // N=   4096, p= 65.98 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, row_major, col_major, 128, 128,  32,  64,  64,  32, 128,   2,   2, false, s, 2); // N=   1024, p= 34.41 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , row_major, row_major,  64, 128,  32,  64,  32,  32, 128,   2,   2, false, s, 0); // N=  16384, p= 47.28 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , row_major, row_major,  64, 128,  32,  64,  32,  32, 128,   2,   2, false, s, 1); // N=   4096, p= 49.13 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , row_major, row_major, 128,  32,  32,  32,  32,  16, 128,   1,   2, false, s, 2); // N=   1024, p= 26.19 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , row_major, row_major,  64, 128,  32,  32,  64,  16, 128,   1,   2, false, s, 0); // N=  16384, p= 30.71 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , row_major, row_major,  64, 128,  32,  32,  64,  16, 128,   1,   2, false, s, 1); // N=   4096, p= 30.80 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , row_major, row_major,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, s, 2); // N=   1024, p= 19.26 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, row_major, row_major,  64, 128,  32,  64,  32,  32, 128,   1,   2, false, s, 0); // N=  16384, p= 70.62 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, row_major, row_major, 128, 128,  32,  64,  64,  32, 128,   1,   2, false, s, 1); // N=   4096, p= 83.73 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, row_major, row_major, 128, 128,  32,  64,  64,  32, 128,   1,   2, false, s, 2); // N=   1024, p= 42.66 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, row_major, row_major, 128, 128,  32,  32,  64,  32, 256,   2,   2, false, s, 0); // N=  16384, p= 65.69 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, row_major, row_major,  64, 128,  32,  32,  64,  16, 128,   2,   2, false, s, 1); // N=   4096, p= 68.57 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, row_major, row_major, 128, 128,  32,  64,  64,  16, 128,   2,   2, false, s, 2); // N=   1024, p= 37.41 [TFlop/s]
#endif
#ifdef COMPILE_CGEMM_KERNEL
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, col_major, 128,  64,  32,  16,  64,  16, 256,   2,   2, false, c, 0); // N=   8192, p= 44.72 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, col_major, 128,  64,  32,  32,  32,  16, 256,   1,   2, false, c, 1); // N=   2048, p= 33.01 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, col_major,  32,  32,  32,  16,  16,  32, 128,   2,   2, false, c, 2); // N=    512, p= 17.99 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, col_major, 128,  64,  32,  16,  64,  16, 256,   1,   2, false, c, 0); // N=   8192, p= 31.25 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, col_major,  64,  64,  32,  16,  64,  16, 128,   2,   2, false, c, 1); // N=   2048, p= 30.15 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, col_major,  32,  32,  32,  16,  16,  32, 128,   1,   2, false, c, 2); // N=    512, p= 18.80 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, col_major, 128,  64,  32,  32,  64,  16, 128,   2,   2, false, c, 0); // N=   8192, p= 96.91 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, col_major, 128,  64,  32,  32,  64,  16, 128,   2,   2, false, c, 1); // N=   2048, p= 89.74 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, col_major,  32, 128,  32,  16,  64,  16, 128,   1,   2, false, c, 2); // N=    512, p= 34.26 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, col_major, 128,  64,  32,  32,  32,  16, 256,   1,   2, false, c, 0); // N=   8192, p= 66.97 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, col_major, 128,  64,  32,  32,  32,  16, 256,   1,   2, false, c, 1); // N=   2048, p= 63.16 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, col_major,  32,  32,  32,  16,  16,  32, 128,   1,   2, false, c, 2); // N=    512, p= 28.00 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, row_major, 128,  64,  32,  32,  32,  16, 256,   2,   2, false, c, 0); // N=   8192, p= 43.02 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, row_major,  64,  64,  32,  16,  64,  16, 128,   1,   2, false, c, 1); // N=   2048, p= 40.02 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, row_major,  32, 128,  32,  32,  32,  16, 128,   1,   2, false, c, 2); // N=    512, p= 19.03 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, row_major, 128,  64,  32,  16,  64,  16, 256,   1,   2, false, c, 0); // N=   8192, p= 31.33 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, row_major,  64,  64,  32,  16,  64,  16, 128,   1,   2, false, c, 1); // N=   2048, p= 29.82 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, row_major,  32,  32,  32,  16,  16,  32, 128,   1,   2, false, c, 2); // N=    512, p= 17.35 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, row_major, 128,  64,  32,  32,  64,  16, 128,   1,   2, false, c, 0); // N=   8192, p= 72.90 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, row_major,  64, 128,  32,  32,  64,  16, 128,   1,   2, false, c, 1); // N=   2048, p= 67.69 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, row_major, 128,  32,  32,  32,  32,  16, 128,   1,   2, false, c, 2); // N=    512, p= 26.09 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, row_major, 128,  64,  32,  32,  32,  16, 256,   1,   2, false, c, 0); // N=   8192, p= 64.71 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, row_major, 128,  64,  32,  32,  32,  16, 256,   1,   2, false, c, 1); // N=   2048, p= 59.59 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, row_major,  32,  32,  32,  16,  16,  32, 128,   1,   2, false, c, 2); // N=    512, p= 26.68 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, conjugate,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 0); // N=   8192, p= 39.55 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, conjugate,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 1); // N=   2048, p= 35.67 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, conjugate,  32,  32,  32,  16,  16,  16, 128,   1,   2, false, c, 2); // N=    512, p= 17.21 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, conjugate,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 0); // N=   8192, p= 28.65 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, conjugate,  64,  32,  32,  16,  32,  32, 128,   1,   2, false, c, 1); // N=   2048, p= 27.00 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, conjugate,  32,  32,  32,  16,  16,  16, 128,   1,   2, false, c, 2); // N=    512, p= 15.22 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, conjugate, 128,  64,  32,  32,  64,  16, 128,   1,   2, false, c, 0); // N=   8192, p= 58.96 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, conjugate, 128,  64,  32,  32,  64,  16, 128,   1,   2, false, c, 1); // N=   2048, p= 49.92 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, conjugate, 128,  32,  32,  32,  32,  16, 128,   1,   2, false, c, 2); // N=    512, p= 20.33 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, conjugate,  64,  64,  32,  32,  32,  32, 128,   1,   2, false, c, 0); // N=   8192, p= 52.34 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, conjugate,  64,  64,  32,  32,  32,  32, 128,   1,   2, false, c, 1); // N=   2048, p= 49.19 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, conjugate,  32,  32,  32,  16,  16,  16, 128,   2,   2, false, c, 2); // N=    512, p= 22.15 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, col_major,  64, 128,  32,  32,  32,  16, 256,   2,   2, false, c, 0); // N=   8192, p= 53.08 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, col_major, 128,  64,  32,  32,  32,  16, 256,   1,   2, false, c, 1); // N=   2048, p= 51.37 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, col_major,  32,  32,  32,  16,  16,  32, 128,   1,   2, false, c, 2); // N=    512, p= 26.95 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, col_major, 128,  64,  32,  16,  64,  16, 256,   1,   2, false, c, 0); // N=   8192, p= 31.35 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, col_major,  64,  64,  32,  16,  64,  16, 128,   1,   2, false, c, 1); // N=   2048, p= 29.93 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, col_major,  32,  32,  32,  16,  16,  32, 128,   1,   2, false, c, 2); // N=    512, p= 18.22 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, col_major,  64, 128,  32,  32,  64,  16, 128,   1,   2, false, c, 0); // N=   8192, p=119.63 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, col_major, 128,  64,  32,  64,  32,  16, 128,   1,   2, false, c, 1); // N=   2048, p=120.24 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, col_major,  64,  64,  32,  64,  16,  16, 128,   1,   2, false, c, 2); // N=    512, p= 43.38 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, col_major,  64, 128,  32,  32,  32,  16, 256,   1,   2, false, c, 0); // N=   8192, p= 64.00 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, col_major, 128,  64,  32,  32,  32,  16, 256,   1,   2, false, c, 1); // N=   2048, p= 59.82 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, col_major,  32,  32,  32,  16,  16,  16, 128,   1,   2, false, c, 2); // N=    512, p= 27.91 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, row_major,  64, 128,  32,  32,  32,  16, 256,   1,   2, false, c, 0); // N=   8192, p= 51.04 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, row_major,  64, 128,  32,  32,  32,  16, 256,   1,   2, false, c, 1); // N=   2048, p= 48.62 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, row_major,  32, 128,  32,  32,  32,  16, 128,   1,   2, false, c, 2); // N=    512, p= 21.84 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, row_major,  64, 128,  32,  16,  64,  16, 256,   1,   2, false, c, 0); // N=   8192, p= 30.84 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, row_major,  64,  64,  32,  16,  64,  16, 128,   1,   2, false, c, 1); // N=   2048, p= 30.09 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, row_major,  32,  32,  32,  16,  16,  32, 128,   1,   2, false, c, 2); // N=    512, p= 17.75 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, row_major,  64, 128,  32,  64,  32,  16, 128,   1,   2, false, c, 0); // N=   8192, p=102.02 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, row_major, 128,  64,  32,  64,  32,  16, 128,   2,   2, false, c, 1); // N=   2048, p= 95.26 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, row_major, 128,  32,  32,  64,  16,  16, 128,   2,   2, false, c, 2); // N=    512, p= 35.24 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, row_major, 128,  64,  32,  32,  32,  16, 256,   1,   2, false, c, 0); // N=   8192, p= 64.55 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, row_major, 128,  64,  32,  32,  32,  16, 256,   1,   2, false, c, 1); // N=   2048, p= 59.86 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, row_major,  32,  32,  32,  16,  16,  32, 128,   1,   2, false, c, 2); // N=    512, p= 26.39 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, conjugate, 128,  64,  32,  32,  32,  16, 256,   1,   2, false, c, 0); // N=   8192, p= 43.86 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, conjugate,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 1); // N=   2048, p= 40.50 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, conjugate,  32,  32,  32,  16,  16,  16, 128,   1,   2, false, c, 2); // N=    512, p= 19.01 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, conjugate,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 0); // N=   8192, p= 28.14 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, conjugate,  64,  32,  32,  16,  32,  32, 128,   1,   2, false, c, 1); // N=   2048, p= 27.00 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, conjugate,  32,  32,  32,  16,  16,  16, 128,   1,   2, false, c, 2); // N=    512, p= 15.12 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, conjugate, 128,  64,  32,  64,  32,  16, 128,   1,   2, false, c, 0); // N=   8192, p= 72.22 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, conjugate,  64,  64,  32,  64,  16,  32, 128,   2,   2, false, c, 1); // N=   2048, p= 63.47 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, conjugate, 128,  32,  32,  32,  16,  16, 256,   2,   2, false, c, 2); // N=    512, p= 26.11 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, conjugate,  64,  64,  32,  32,  32,  32, 128,   1,   2, false, c, 0); // N=   8192, p= 51.91 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, conjugate,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 1); // N=   2048, p= 48.95 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, conjugate,  32,  32,  32,  16,  16,  16, 128,   2,   2, false, c, 2); // N=    512, p= 22.22 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, col_major,  64,  64,  32,  64,  16,  16, 128,   2,   2, false, c, 0); // N=   8192, p= 47.33 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, col_major,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 1); // N=   2048, p= 43.46 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, col_major,  32,  32,  32,  16,  16,  32, 128,   2,   2, false, c, 2); // N=    512, p= 21.75 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, col_major,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 0); // N=   8192, p= 28.64 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, col_major,  32,  64,  32,  16,  32,  32, 128,   1,   2, false, c, 1); // N=   2048, p= 27.08 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, col_major,  32,  32,  32,  16,  16,  16, 128,   2,   2, false, c, 2); // N=    512, p= 16.00 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, col_major,  64, 128,  32,  64,  32,  16, 128,   1,   2, false, c, 0); // N=   8192, p= 86.71 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, col_major,  64,  64,  32,  32,  16,  16, 256,   2,   2, false, c, 1); // N=   2048, p= 81.54 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, col_major,  32, 128,  32,  32,  16,  16, 256,   1,   2, false, c, 2); // N=    512, p= 31.99 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, col_major,  64,  64,  32,  32,  16,  16, 256,   1,   2, false, c, 0); // N=   8192, p= 52.83 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, col_major,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 1); // N=   2048, p= 50.00 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, col_major,  32,  32,  32,  16,  16,  16, 128,   2,   2, false, c, 2); // N=    512, p= 23.00 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, row_major,  64,  64,  32,  64,  16,  16, 128,   2,   2, false, c, 0); // N=   8192, p= 45.84 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, row_major,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 1); // N=   2048, p= 42.46 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, row_major,  32,  32,  32,  16,  16,  16, 128,   2,   2, false, c, 2); // N=    512, p= 19.69 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, row_major,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 0); // N=   8192, p= 28.75 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, row_major,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 1); // N=   2048, p= 26.71 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, row_major,  32,  32,  32,  16,  16,  16, 128,   1,   2, false, c, 2); // N=    512, p= 15.81 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, row_major,  64, 128,  32,  64,  32,  16, 128,   1,   2, false, c, 0); // N=   8192, p= 74.84 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, row_major,  64,  64,  32,  64,  16,  32, 128,   1,   2, false, c, 1); // N=   2048, p= 66.78 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, row_major,  32, 128,  32,  32,  16,  16, 256,   1,   2, false, c, 2); // N=    512, p= 26.72 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, row_major,  64,  64,  32,  32,  16,  16, 256,   1,   2, false, c, 0); // N=   8192, p= 52.14 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, row_major,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 1); // N=   2048, p= 49.44 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, row_major,  32,  32,  32,  16,  16,  16, 128,   2,   2, false, c, 2); // N=    512, p= 22.97 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, conjugate,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 0); // N=   8192, p= 41.75 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, conjugate,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 1); // N=   2048, p= 39.53 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, conjugate,  32,  32,  32,  16,  16,  16, 128,   1,   2, false, c, 2); // N=    512, p= 16.93 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, conjugate,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 0); // N=   8192, p= 26.38 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, conjugate,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 1); // N=   2048, p= 24.12 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, conjugate,  32,  32,  32,  16,  16,  16, 128,   1,   2, false, c, 2); // N=    512, p= 14.00 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, conjugate,  64,  64,  32,  64,  16,  32, 128,   2,   2, false, c, 0); // N=   8192, p= 61.38 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, conjugate,  64,  64,  32,  64,  16,  32, 128,   2,   2, false, c, 1); // N=   2048, p= 57.45 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, conjugate,  32,  32,  32,  16,  16,  16, 128,   1,   2, false, c, 2); // N=    512, p= 21.49 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, conjugate,  64,  64,  32,  32,  16,  16, 256,   1,   2, false, c, 0); // N=   8192, p= 49.81 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, conjugate,  64,  64,  32,  32,  16,  16, 256,   1,   2, false, c, 1); // N=   2048, p= 45.85 [TFlop/s]
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, conjugate,  32,  32,  32,  16,  16,  16, 128,   1,   2, false, c, 2); // N=    512, p= 19.08 [TFlop/s]
#endif

#ifdef COMPILE_SGEMM_STRIDEDBATCH_KERNEL
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , col_major, col_major, 128,  64,  32,  32,  64,  32, 128,   1,   2, false, s, 0); // N=   1024, p= 44.68 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , col_major, col_major, 128,  64,  32,  32,  64,  32, 128,   1,   2, false, s, 1); // N=    256, p= 24.63 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , col_major, col_major,  64,  64,  64,  32,  32,  64, 128,   1,   2, false, s, 2); // N=     64, p=  6.38 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , col_major, col_major, 128,  64,  32,  64,  32,  16, 128,   1,   2, false, s, 0); // N=   1024, p= 27.86 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , col_major, col_major, 128,  64,  32,  64,  32,  16, 128,   1,   2, false, s, 1); // N=    256, p= 21.60 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , col_major, col_major,  64,  64,  64,  32,  32,  32, 128,   1,   2, false, s, 2); // N=     64, p=  7.33 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, col_major, col_major, 128, 128,  32,  64,  64,  32, 128,   2,   2, false, s, 0); // N=   1024, p= 85.03 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, col_major, col_major, 128,  64,  32,  32,  64,  32, 128,   1,   2, false, s, 1); // N=    256, p= 41.12 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, col_major, col_major,  64,  64,  64,  32,  32,  64, 128,   1,   2, false, s, 2); // N=     64, p=  9.22 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, col_major, col_major, 128, 128,  32,  64,  64,  16, 128,   2,   2, false, s, 0); // N=   1024, p= 67.49 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, col_major, col_major, 128,  64,  32,  32,  64,  32, 128,   1,   2, false, s, 1); // N=    256, p= 39.24 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, col_major, col_major,  64,  64,  64,  32,  32,  64, 128,   1,   2, false, s, 2); // N=     64, p=  9.17 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , col_major, row_major, 128,  64,  32,  64,  32,  32, 128,   1,   2, false, s, 0); // N=   1024, p= 38.41 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , col_major, row_major, 128,  64,  32,  64,  32,  32, 128,   1,   2, false, s, 1); // N=    256, p= 26.91 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , col_major, row_major,  64,  64,  64,  32,  32,  64, 128,   1,   2, false, s, 2); // N=     64, p=  8.14 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , col_major, row_major, 128,  64,  32,  32,  64,  16, 128,   2,   2, false, s, 0); // N=   1024, p= 28.10 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , col_major, row_major, 128,  64,  32,  32,  64,  16, 128,   1,   2, false, s, 1); // N=    256, p= 21.80 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , col_major, row_major,  64,  64,  64,  32,  32,  32, 128,   1,   2, false, s, 2); // N=     64, p=  7.39 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, col_major, row_major, 128, 128,  32,  64,  64,  32, 128,   2,   2, false, s, 0); // N=   1024, p= 76.12 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, col_major, row_major, 128, 128,  32,  64,  64,  16, 128,   1,   2, false, s, 1); // N=    256, p= 39.12 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, col_major, row_major,  64,  64,  64,  32,  32,  64, 128,   1,   2, false, s, 2); // N=     64, p=  9.02 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, col_major, row_major, 128, 128,  32,  64,  64,  32, 128,   1,   2, false, s, 0); // N=   1024, p= 68.45 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, col_major, row_major, 128,  64,  32,  64,  32,  32, 128,   1,   2, false, s, 1); // N=    256, p= 40.95 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, col_major, row_major,  64,  64,  64,  32,  32,  64, 128,   1,   2, false, s, 2); // N=     64, p=  9.17 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , row_major, col_major, 128,  64,  32,  64,  32,  32, 128,   1,   2, false, s, 0); // N=   1024, p= 47.68 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , row_major, col_major, 128,  64,  32,  64,  32,  32, 128,   1,   2, false, s, 1); // N=    256, p= 33.01 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , row_major, col_major,  64,  64,  64,  32,  32,  64, 128,   1,   2, false, s, 2); // N=     64, p=  8.47 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , row_major, col_major, 128,  64,  32,  32,  64,  16, 128,   1,   2, false, s, 0); // N=   1024, p= 26.66 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , row_major, col_major, 128,  64,  32,  32,  64,  16, 128,   1,   2, false, s, 1); // N=    256, p= 21.10 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , row_major, col_major,  64,  64,  64,  32,  32,  32, 128,   1,   2, false, s, 2); // N=     64, p=  7.17 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, row_major, col_major, 128, 128,  32,  64,  64,  32, 128,   2,   2, false, s, 0); // N=   1024, p= 90.90 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, row_major, col_major, 128,  64,  32,  32,  32,  32, 256,   1,   2, false, s, 1); // N=    256, p= 41.68 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, row_major, col_major,  64,  64,  64,  32,  32,  64, 128,   1,   2, false, s, 2); // N=     64, p=  9.64 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, row_major, col_major, 128, 128,  32,  64,  64,  32, 128,   2,   2, false, s, 0); // N=   1024, p= 62.73 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, row_major, col_major, 128,  64,  32,  32,  64,  16, 128,   1,   2, false, s, 1); // N=    256, p= 38.33 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, row_major, col_major,  64,  64,  64,  32,  32,  64, 128,   1,   2, false, s, 2); // N=     64, p=  8.78 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , row_major, row_major, 128,  64,  32,  64,  32,  32, 128,   2,   2, false, s, 0); // N=   1024, p= 45.04 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , row_major, row_major, 128,  64,  32,  64,  32,  32, 128,   1,   2, false, s, 1); // N=    256, p= 30.68 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , row_major, row_major,  64,  64,  64,  32,  32,  64, 128,   1,   2, false, s, 2); // N=     64, p=  8.18 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , row_major, row_major, 128,  64,  32,  32,  64,  16, 128,   1,   2, false, s, 0); // N=   1024, p= 27.84 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , row_major, row_major, 128,  64,  32,  32,  64,  16, 128,   1,   2, false, s, 1); // N=    256, p= 21.88 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , row_major, row_major,  64,  64,  64,  32,  32,  32, 128,   1,   2, false, s, 2); // N=     64, p=  7.36 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, row_major, row_major, 128, 128,  32,  64,  64,  32, 128,   2,   2, false, s, 0); // N=   1024, p= 83.69 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, row_major, row_major, 128,  64,  32,  64,  32,  32, 128,   2,   2, false, s, 1); // N=    256, p= 40.49 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, row_major, row_major,  64,  64,  64,  32,  32,  64, 128,   1,   2, false, s, 2); // N=     64, p=  9.37 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, row_major, row_major, 128, 128,  32,  64,  64,  16, 128,   1,   2, false, s, 0); // N=   1024, p= 65.96 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, row_major, row_major, 128,  64,  32,  32,  64,  32, 128,   1,   2, false, s, 1); // N=    256, p= 39.27 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, row_major, row_major,  64,  64,  64,  32,  32,  64, 128,   1,   2, false, s, 2); // N=     64, p=  9.17 [TFlop/s]
#endif
#ifdef COMPILE_CGEMM_STRIDEDBATCH_KERNEL
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, col_major,  64,  64,  32,  16,  64,  32, 128,   1,   2, false, c, 0); // N=   1024, p= 44.93 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, col_major,  64,  64,  32,  16,  64,  32, 128,   1,   2, false, c, 1); // N=    256, p= 30.66 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, col_major,  64,  32,  32,  16,  32,  32, 128,   2,   2, false, c, 2); // N=     64, p= 14.31 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, col_major,  64,  64,  32,  16,  64,  16, 128,   1,   2, false, c, 0); // N=   1024, p= 30.79 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, col_major,  64,  32,  32,  16,  32,  32, 128,   1,   2, false, c, 1); // N=    256, p= 24.96 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, col_major,  64,  32,  32,  16,  32,  16, 128,   2,   2, false, c, 2); // N=     64, p= 15.43 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, col_major, 128,  64,  32,  32,  64,  16, 128,   1,   2, false, c, 0); // N=   1024, p= 88.76 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, col_major, 128,  64,  32,  32,  64,  16, 128,   1,   2, false, c, 1); // N=    256, p= 64.39 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, col_major,  64,  64,  64,  16,  64,  32, 128,   1,   2, false, c, 2); // N=     64, p= 24.05 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, col_major, 128,  64,  32,  32,  32,  16, 256,   1,   2, false, c, 0); // N=   1024, p= 61.04 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, col_major, 128,  64,  32,  32,  32,  16, 256,   1,   2, false, c, 1); // N=    256, p= 46.78 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, col_major,  64,  64,  64,  32,  16,  32, 256,   1,   2, false, c, 2); // N=     64, p= 20.34 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, row_major,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 0); // N=   1024, p= 41.29 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, row_major,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 1); // N=    256, p= 32.49 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, row_major,  64,  64,  64,  32,  32,  16, 128,   1,   2, false, c, 2); // N=     64, p= 17.57 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, row_major,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 0); // N=   1024, p= 28.77 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, row_major,  64,  32,  32,  16,  32,  32, 128,   2,   2, false, c, 1); // N=    256, p= 24.71 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, row_major,  64,  32,  32,  16,  32,  16, 128,   1,   2, false, c, 2); // N=     64, p= 15.32 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, row_major, 128,  64,  32,  32,  64,  16, 128,   1,   2, false, c, 0); // N=   1024, p= 67.23 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, row_major, 128,  64,  32,  32,  64,  16, 128,   1,   2, false, c, 1); // N=    256, p= 50.76 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, row_major,  64,  64,  64,  32,  32,  16, 128,   1,   2, false, c, 2); // N=     64, p= 21.30 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, row_major,  64,  64,  32,  32,  32,  32, 128,   2,   2, false, c, 0); // N=   1024, p= 57.31 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, row_major,  64,  64,  32,  32,  32,  32, 128,   2,   2, false, c, 1); // N=    256, p= 43.23 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, row_major,  64,  64,  64,  32,  16,  32, 256,   1,   2, false, c, 2); // N=     64, p= 20.34 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, conjugate,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 0); // N=   1024, p= 38.97 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, conjugate,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 1); // N=    256, p= 27.73 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, conjugate,  64,  32,  32,  32,  16,  16, 128,   1,   2, false, c, 2); // N=     64, p= 15.50 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, conjugate,  64,  32,  32,  16,  32,  32, 128,   1,   2, false, c, 0); // N=   1024, p= 27.11 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, conjugate,  64,  32,  32,  16,  32,  32, 128,   1,   2, false, c, 1); // N=    256, p= 22.45 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, conjugate,  64,  64,  64,  16,  32,  16, 256,   2,   2, false, c, 2); // N=     64, p= 14.28 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, conjugate, 128, 128,  32,  32,  64,  16, 256,   1,   2, false, c, 0); // N=   1024, p= 59.50 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, conjugate, 128, 128,  32,  32,  64,  16, 256,   1,   2, false, c, 1); // N=    256, p= 36.40 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, conjugate,  64,  64,  64,  32,  32,  16, 128,   1,   2, false, c, 2); // N=     64, p= 17.21 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, conjugate,  64,  64,  32,  32,  32,  32, 128,   1,   2, false, c, 0); // N=   1024, p= 52.08 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, conjugate,  64,  64,  32,  32,  16,  16, 256,   2,   2, false, c, 1); // N=    256, p= 37.48 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, conjugate,  64,  64,  64,  32,  16,  32, 256,   2,   2, false, c, 2); // N=     64, p= 18.75 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, col_major,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 0); // N=   1024, p= 51.34 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, col_major,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 1); // N=    256, p= 39.90 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, col_major,  64,  32,  32,  32,  16,  32, 128,   2,   2, false, c, 2); // N=     64, p= 19.74 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, col_major,  64,  64,  32,  16,  64,  16, 128,   1,   2, false, c, 0); // N=   1024, p= 31.33 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, col_major,  64,  64,  32,  16,  64,  16, 128,   1,   2, false, c, 1); // N=    256, p= 25.08 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, col_major,  64,  64,  64,  16,  32,  16, 256,   2,   2, false, c, 2); // N=     64, p= 15.43 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, col_major, 128,  64,  32,  64,  32,  16, 128,   1,   2, false, c, 0); // N=   1024, p=119.62 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, col_major, 128,  64,  32,  32,  64,  16, 128,   1,   2, false, c, 1); // N=    256, p= 77.28 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, col_major,  64,  64,  64,  64,  16,  32, 128,   2,   2, false, c, 2); // N=     64, p= 25.91 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, col_major, 128,  64,  32,  32,  32,  16, 256,   1,   2, false, c, 0); // N=   1024, p= 59.02 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, col_major, 128,  64,  32,  32,  32,  16, 256,   1,   2, false, c, 1); // N=    256, p= 45.77 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, col_major,  64,  32,  32,  32,  16,  16, 128,   1,   2, false, c, 2); // N=     64, p= 20.46 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, row_major,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 0); // N=   1024, p= 47.92 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, row_major,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 1); // N=    256, p= 38.19 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, row_major,  64,  64,  64,  32,  32,  16, 128,   2,   2, false, c, 2); // N=     64, p= 19.17 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, row_major,  64,  64,  32,  16,  64,  16, 128,   1,   2, false, c, 0); // N=   1024, p= 31.18 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, row_major,  64,  64,  32,  16,  64,  16, 128,   1,   2, false, c, 1); // N=    256, p= 24.90 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, row_major,  64,  32,  32,  16,  32,  16, 128,   2,   2, false, c, 2); // N=     64, p= 15.43 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, row_major, 128,  64,  32,  64,  32,  16, 128,   2,   2, false, c, 0); // N=   1024, p= 90.58 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, row_major, 128,  64,  32,  64,  32,  32, 128,   1,   2, false, c, 1); // N=    256, p= 65.40 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, row_major,  64,  64,  64,  64,  16,  32, 128,   2,   2, false, c, 2); // N=     64, p= 24.67 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, row_major, 128,  64,  32,  32,  32,  16, 256,   1,   2, false, c, 0); // N=   1024, p= 58.91 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, row_major, 128,  64,  32,  32,  32,  16, 256,   1,   2, false, c, 1); // N=    256, p= 45.61 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, row_major,  64,  32,  32,  32,  16,  16, 128,   1,   2, false, c, 2); // N=     64, p= 20.46 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, conjugate,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 0); // N=   1024, p= 44.86 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, conjugate, 128,  64,  32,  32,  32,  16, 256,   1,   2, false, c, 1); // N=    256, p= 31.60 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, conjugate,  64,  32,  32,  32,  16,  32, 128,   1,   2, false, c, 2); // N=     64, p= 17.03 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, conjugate,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 0); // N=   1024, p= 27.46 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, conjugate,  64,  32,  32,  16,  32,  32, 128,   1,   2, false, c, 1); // N=    256, p= 22.66 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, conjugate,  64,  64,  64,  16,  32,  16, 256,   2,   2, false, c, 2); // N=     64, p= 13.89 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, conjugate, 128, 128,  32,  64,  32,  16, 256,   1,   2, false, c, 0); // N=   1024, p= 76.47 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, conjugate, 128,  64,  32,  64,  16,  16, 256,   1,   2, false, c, 1); // N=    256, p= 46.43 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, conjugate,  64,  64,  64,  32,  16,  16, 256,   2,   2, false, c, 2); // N=     64, p= 20.84 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, conjugate,  64,  64,  32,  32,  32,  32, 128,   1,   2, false, c, 0); // N=   1024, p= 52.63 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, conjugate, 128,  64,  32,  32,  32,  16, 256,   1,   2, false, c, 1); // N=    256, p= 36.62 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, conjugate,  64,  64,  64,  32,  16,  64, 256,   1,   2, false, c, 2); // N=     64, p= 18.44 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, col_major,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 0); // N=   1024, p= 47.84 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, col_major,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 1); // N=    256, p= 32.18 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, col_major,  64,  64,  64,  32,  16,  16, 256,   2,   2, false, c, 2); // N=     64, p= 17.61 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, col_major,  64,  64,  32,  16,  64,  16, 128,   1,   2, false, c, 0); // N=   1024, p= 29.00 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, col_major,  64,  64,  32,  16,  64,  16, 128,   1,   2, false, c, 1); // N=    256, p= 21.77 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, col_major,  64,  64,  64,  16,  32,  16, 256,   2,   2, false, c, 2); // N=     64, p= 14.19 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, col_major, 128, 128,  32,  64,  32,  16, 256,   2,   2, false, c, 0); // N=   1024, p= 95.88 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, col_major, 128, 128,  32,  64,  32,  16, 256,   1,   2, false, c, 1); // N=    256, p= 47.06 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, col_major,  64,  64,  64,  32,  16,  16, 256,   2,   2, false, c, 2); // N=     64, p= 22.98 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, col_major,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 0); // N=   1024, p= 53.66 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, col_major,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 1); // N=    256, p= 34.70 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, col_major,  64,  64,  64,  32,  16,  64, 256,   1,   2, false, c, 2); // N=     64, p= 18.34 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, row_major,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 0); // N=   1024, p= 45.18 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, row_major,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 1); // N=    256, p= 30.69 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, row_major,  64,  64,  64,  16,  32,  16, 256,   2,   2, false, c, 2); // N=     64, p= 17.16 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, row_major,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 0); // N=   1024, p= 27.39 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, row_major,  32,  32,  32,  16,  16,  16, 128,   1,   2, false, c, 1); // N=    256, p= 20.94 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, row_major,  64,  64,  64,  16,  32,  16, 256,   1,   2, false, c, 2); // N=     64, p= 14.28 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, row_major, 128, 128,  32,  64,  32,  16, 256,   1,   2, false, c, 0); // N=   1024, p= 78.89 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, row_major, 128, 128,  32,  64,  32,  16, 256,   1,   2, false, c, 1); // N=    256, p= 41.69 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, row_major,  64,  64,  64,  32,  16,  16, 256,   2,   2, false, c, 2); // N=     64, p= 20.84 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, row_major,  64,  64,  32,  32,  32,  32, 128,   1,   2, false, c, 0); // N=   1024, p= 53.13 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, row_major,  64,  64,  32,  32,  32,  32, 128,   1,   2, false, c, 1); // N=    256, p= 34.32 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, row_major,  64,  64,  64,  32,  16,  64, 256,   2,   2, false, c, 2); // N=     64, p= 18.49 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, conjugate,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 0); // N=   1024, p= 41.35 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, conjugate,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 1); // N=    256, p= 26.40 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, conjugate,  64,  64,  64,  32,  16,  32, 256,   2,   2, false, c, 2); // N=     64, p= 15.90 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, conjugate,  64,  64,  32,  32,  32,  16, 128,   1,   2, false, c, 0); // N=   1024, p= 25.68 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, conjugate,  32,  32,  32,  16,  16,  16, 128,   1,   2, false, c, 1); // N=    256, p= 18.68 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, conjugate,  64,  64,  64,  16,  32,  16, 256,   1,   2, false, c, 2); // N=     64, p= 13.29 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, conjugate, 128, 128,  32,  64,  32,  16, 256,   1,   2, false, c, 0); // N=   1024, p= 70.08 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, conjugate, 128, 128,  32,  64,  32,  16, 256,   1,   2, false, c, 1); // N=    256, p= 37.41 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, conjugate,  64,  64,  64,  32,  16,  16, 256,   2,   2, false, c, 2); // N=     64, p= 19.07 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, conjugate,  64,  64,  32,  32,  32,  16, 128,   2,   2, false, c, 0); // N=   1024, p= 47.19 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, conjugate,  32,  32,  32,  16,  16,  16, 128,   1,   2, false, c, 1); // N=    256, p= 28.63 [TFlop/s]
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, conjugate,  64,  32,  32,  16,  16,  16, 256,   1,   2, false, c, 2); // N=     64, p= 17.48 [TFlop/s]
#endif

#else // ENABLE_A100_OPTIMAZED_PARAMETERS
#ifdef COMPILE_SGEMM_KERNEL
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , col_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , col_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , col_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , col_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , col_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , col_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, col_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, col_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, col_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, col_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, col_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, col_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , col_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , col_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , col_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , col_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , col_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , col_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, col_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, col_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, col_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, col_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, col_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, col_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , row_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , row_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , row_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , row_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , row_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , row_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, row_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, row_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, row_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, row_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, row_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, row_major, col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , row_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , row_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , row_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , row_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , row_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , row_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, row_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, row_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, row_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, row_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, row_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, row_major, row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
#endif
#ifdef COMPILE_CGEMM_KERNEL
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 32, 128, 1, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
#endif

#ifdef COMPILE_SGEMM_STRIDEDBATCH_KERNEL
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 2); // Not optimized but works on any Ampere GPUs
#endif
#ifdef COMPILE_CGEMM_STRIDEDBATCH_KERNEL
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1); // Not optimized but works on any Ampere GPUs
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 2); // Not optimized but works on any Ampere GPUs
#endif
#endif

	(*handle)->exp_stats_enabled = false;
	(*handle)->buffer_length = 10000;
	(*handle)->ignore_threshold = 0;
	(*handle)->lost_threshold = 0;
	(*handle)->current_buffer_id = 1;
	(*handle)->counter_init_disabled = false;
	CUTF_CHECK_ERROR(cudaMalloc    (&((*handle)->dev_lost_counter_buffer ), sizeof(cumpsgemm::counter_t) * (*handle)->buffer_length));
	CUTF_CHECK_ERROR(cudaMalloc    (&((*handle)->dev_total_counter_buffer), sizeof(cumpsgemm::counter_t) * (*handle)->buffer_length));
	CUTF_CHECK_ERROR(cudaMallocHost(&((*handle)->host_lost_counter_buffer ), sizeof(cumpsgemm::counter_t) * (*handle)->buffer_length));
	CUTF_CHECK_ERROR(cudaMallocHost(&((*handle)->host_total_counter_buffer), sizeof(cumpsgemm::counter_t) * (*handle)->buffer_length));

	(*handle)->host_lost_counter_buffer [0] = 1;
	(*handle)->host_total_counter_buffer[0] = 1;

	return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cuMpSGEMM_destroy(cuMpSGEMM_handle_t handle) {
	CUTF_CHECK_ERROR(cudaFree    (handle->dev_lost_counter_buffer ));
	CUTF_CHECK_ERROR(cudaFree    (handle->dev_total_counter_buffer));
	CUTF_CHECK_ERROR(cudaFreeHost(handle->host_lost_counter_buffer ));
	CUTF_CHECK_ERROR(cudaFreeHost(handle->host_total_counter_buffer));

	delete handle;
	return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cuMpSGEMM_set_stream(cuMpSGEMM_handle_t handle, const cudaStream_t cuda_stream) {
	handle->cuda_stream = cuda_stream;
	return CUBLAS_STATUS_SUCCESS;
}
}
