#ifndef __CUMPSGEMM_HPP__
#define __CUMPSGEMM_HPP__
#include "cumpsgemm.h"

namespace cumpsgemm {
template <class T>
cublasStatus_t gemm(
		cuMpSGEMM_handle_t handle,
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const uint64_t m,
		const uint64_t n,
		const uint64_t k,
		const T* alpha,
		const T* const a_dmem_ptr, const uint64_t lda,
		const T* const b_dmem_ptr, const uint64_t ldb,
		const T* beta,
		T* const c_dmem_ptr, const uint64_t ldc,
		const cuMpSGEMM_compute_mode_t compute_mode,
		cudaStream_t cuda_stream = 0
		);
template <class T>
cublasStatus_t gemm_stridedBatch(
		cuMpSGEMM_handle_t handle,
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const uint64_t m,
		const uint64_t n,
		const uint64_t k,
		const T* alpha,
		const T* const a_dmem_ptr, const uint64_t lda, const uint64_t stridea,
		const T* const b_dmem_ptr, const uint64_t ldb, const uint64_t strideb,
		const T* beta,
		T* const c_dmem_ptr, const uint64_t ldc, const uint64_t stridec,
		const uint64_t batch_count,
		const cuMpSGEMM_compute_mode_t compute_mode,
		cudaStream_t cuda_stream = 0
		);
} // namespace cumpsgemm
#endif
