#ifndef __CUMPSGEMM_H__
#define __CUMPSGEMM_H__
#include <cublas_v2.h>

enum cuMpSGEMM_compute_mode_t {
	CUMPSGEMM_CUBLAS   = 0,
	CUMPSGEMM_FP16TCEC = 1,
	CUMPSGEMM_TF32TCEC = 2,
	CUMPSGEMM_FP16TC   = 3,
	CUMPSGEMM_TF32TC   = 4,
};

const char* cuMpSGEMM_get_compute_mode_string (
		const cuMpSGEMM_compute_mode_t mode
		);

// User defined function
extern "C" cuMpSGEMM_compute_mode_t cuMpSGEMM_get_compute_mode (
		const char* const func_name,
		cublasHandle_t const cublas_handle,
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const unsigned m, const unsigned n, const unsigned k
		);

extern "C" int cuMpSGEMM_sgemm(
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const uint64_t m,
		const uint64_t n,
		const uint64_t k,
		const float* alpha,
		const float* const a_dmem_ptr, const uint64_t lda,
		const float* const b_dmem_ptr, const uint64_t ldb,
		const float* beta,
		const float* const c_dmem_ptr, const uint64_t ldc
		);

extern "C" int cuMpSGEMM_cgemm(
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const uint64_t m,
		const uint64_t n,
		const uint64_t k,
		const float* alpha,
		const float* const a_dmem_ptr, const uint64_t lda,
		const float* const b_dmem_ptr, const uint64_t ldb,
		const float* beta,
		const float* const c_dmem_ptr, const uint64_t ldc
		);

#endif
