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

extern "C" {
cublasStatus_t cuMpSGEMM_create(cuMpSGEMM_handle_t* const handle) {
	if ((*handle = new cuMpSGEMM_handle) == nullptr) {
		return CUBLAS_STATUS_INTERNAL_ERROR;
	}

	int num_sms;
	cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
	(*handle)->num_sms = num_sms;

	using tf32 = nvcuda::wmma::precision::tf32;

	// set kernel modules
#ifdef COMPILE_SGEMM_KERNEL
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , col_major, col_major, 64 , 128, 32, 32, 64, 32, 128, 2, 2, false, s, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , col_major, col_major, 64 , 64 , 32, 32, 32, 32, 128, 1, 2, false, s, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , col_major, col_major, 64 , 64 , 32, 32, 32, 32, 128, 1, 2, false, s, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , col_major, col_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, col_major, col_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, col_major, col_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, col_major, col_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, col_major, col_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , row_major, col_major, 64 , 128, 32, 32, 64, 32, 128, 1, 2, false, s, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , row_major, col_major, 64 , 64 , 32, 32, 32, 32, 128, 1, 2, false, s, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , row_major, col_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , row_major, col_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, row_major, col_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, row_major, col_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, row_major, col_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, row_major, col_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , col_major, row_major, 64 , 128, 32, 64, 32, 32, 128, 1, 2, false, s, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , col_major, row_major, 64 , 64 , 64, 32, 32, 32, 128, 2, 2, false, s, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , col_major, row_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , col_major, row_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, col_major, row_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, col_major, row_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, col_major, row_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, col_major, row_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , row_major, row_major, 64 , 128, 32, 64, 32, 32, 128, 1, 2, false, s, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, with_ec   , row_major, row_major, 128, 64 , 32, 64, 32, 32, 128, 1, 2, false, s, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , row_major, row_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, with_ec   , row_major, row_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, row_major, row_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , half, without_ec, row_major, row_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, row_major, row_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, float    , tf32, without_ec, row_major, row_major, 64 , 64 , 32, 32, 32, 16, 128, 2, 2, false, s, 1);
#endif
#ifdef COMPILE_CGEMM_KERNEL
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, half, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_KERNEL_MODULE((*handle)->gemm_module, cuComplex, tf32, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
#endif

#ifdef COMPILE_SGEMM_STRIDEDBATCH_KERNEL
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , half, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, float    , tf32, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, s, 1);
#endif
#ifdef COMPILE_CGEMM_STRIDEDBATCH_KERNEL
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, col_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, row_major, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, col_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, row_major, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, with_ec   , conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, half, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 0);
	SET_GEMM_STRIDEDBATCH_KERNEL_MODULE((*handle)->gemm_stridedBatch_module, cuComplex, tf32, without_ec, conjugate, conjugate, 64, 64, 32, 32, 32, 16, 128, 2, 2, false, c, 1);
#endif

	return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cuMpSGEMM_destroy(cuMpSGEMM_handle_t handle) {
	delete handle;
	return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cuMpSGEMM_set_stream(cuMpSGEMM_handle_t handle, const cudaStream_t cuda_stream) {
	handle->cuda_stream = cuda_stream;
	return CUBLAS_STATUS_SUCCESS;
}
}
