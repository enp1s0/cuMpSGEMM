#ifndef __CUMPSGEMM_HPP__
#define __CUMPSGEMM_HPP__
#include "cumpsgemm.h"

namespace cumpsgemm {
template <class T>
cublasStatus_t gemm(
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
} // namespace cumpsgemm
#endif
