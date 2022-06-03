#include <iostream>
#include <cassert>
#include <cublas.h>
#include <cumpsgemm/cumpsgemm.hpp>

#include "cumpsgemm_internal.hpp"

// cuMpGEMM implementation

namespace {
template <
	class T,
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned BLOCK_SIZE,
	class TC_T,
	class EC
>
void layout_selector (
			const cublasOperation_t op_A,
			const cublasOperation_t op_B,
			const std::size_t m,
			const std::size_t n,
			const std::size_t k,
			const T alpha,
			const T* const a_ptr, const std::size_t lda,
			const T* const b_ptr, const std::size_t ldb,
			const T beta,
			T* const c_ptr, const std::size_t ldc,
			cudaStream_t cuda_stream
		) {
#define CASE(A, a, B, b) \
	if (op_A == a && op_B == b) {cumpsgemm::launch_kernel<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, A, B, TC_T, EC>(m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc, cuda_stream);return;}

	CASE(cumpsgemm::col_major, CUBLAS_OP_N, cumpsgemm::col_major, CUBLAS_OP_N);
	CASE(cumpsgemm::row_major, CUBLAS_OP_T, cumpsgemm::col_major, CUBLAS_OP_N);
	CASE(cumpsgemm::conjugate, CUBLAS_OP_C, cumpsgemm::col_major, CUBLAS_OP_N);
	CASE(cumpsgemm::col_major, CUBLAS_OP_N, cumpsgemm::row_major, CUBLAS_OP_T);
	CASE(cumpsgemm::row_major, CUBLAS_OP_T, cumpsgemm::row_major, CUBLAS_OP_T);
	CASE(cumpsgemm::conjugate, CUBLAS_OP_C, cumpsgemm::row_major, CUBLAS_OP_T);
	CASE(cumpsgemm::col_major, CUBLAS_OP_N, cumpsgemm::conjugate, CUBLAS_OP_C);
	CASE(cumpsgemm::row_major, CUBLAS_OP_T, cumpsgemm::conjugate, CUBLAS_OP_C);
	CASE(cumpsgemm::conjugate, CUBLAS_OP_C, cumpsgemm::conjugate, CUBLAS_OP_C);
#undef CASE
}

template <
	class T,
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned BLOCK_SIZE,
	class TC_T,
	class EC
>
void stridedBatch_layout_selector (
			const cublasOperation_t op_A,
			const cublasOperation_t op_B,
			const std::size_t m,
			const std::size_t n,
			const std::size_t k,
			const T alpha,
			const T* const a_ptr, const std::size_t lda, const std::size_t strideda,
			const T* const b_ptr, const std::size_t ldb, const std::size_t stridedb,
			const T beta,
			T* const c_ptr, const std::size_t ldc, const std::size_t stridedc,
			const std::size_t batch_count,
			cudaStream_t cuda_stream
		) {
#define CASE(A, a, B, b) \
	if (op_A == a && op_B == b) {cumpsgemm::launch_stridedBatch_kernel<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, A, B, TC_T, EC>(m, n, k, alpha, a_ptr, lda, strideda, b_ptr, ldb, stridedb, beta, c_ptr, ldc, stridedc, batch_count, cuda_stream);return;}

	CASE(cumpsgemm::col_major, CUBLAS_OP_N, cumpsgemm::col_major, CUBLAS_OP_N);
	CASE(cumpsgemm::row_major, CUBLAS_OP_T, cumpsgemm::col_major, CUBLAS_OP_N);
	CASE(cumpsgemm::conjugate, CUBLAS_OP_C, cumpsgemm::col_major, CUBLAS_OP_N);
	CASE(cumpsgemm::col_major, CUBLAS_OP_N, cumpsgemm::row_major, CUBLAS_OP_T);
	CASE(cumpsgemm::row_major, CUBLAS_OP_T, cumpsgemm::row_major, CUBLAS_OP_T);
	CASE(cumpsgemm::conjugate, CUBLAS_OP_C, cumpsgemm::row_major, CUBLAS_OP_T);
	CASE(cumpsgemm::col_major, CUBLAS_OP_N, cumpsgemm::conjugate, CUBLAS_OP_C);
	CASE(cumpsgemm::row_major, CUBLAS_OP_T, cumpsgemm::conjugate, CUBLAS_OP_C);
	CASE(cumpsgemm::conjugate, CUBLAS_OP_C, cumpsgemm::conjugate, CUBLAS_OP_C);
#undef CASE
}
} // unnamed namespace

template <class T>
cublasStatus_t cumpsgemm::gemm(
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
		cudaStream_t cuda_stream
		) {
	switch (compute_mode) {
	case CUMPSGEMM_FP16TC:   layout_selector<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, half                         , mtk::wmma::tcec::without_ec>(op_A, op_B, m, n, k, *alpha, a_dmem_ptr, lda, b_dmem_ptr, ldb, *beta, c_dmem_ptr, ldc, cuda_stream);break;
	case CUMPSGEMM_FP16TCEC: layout_selector<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, half                         , mtk::wmma::tcec::with_ec   >(op_A, op_B, m, n, k, *alpha, a_dmem_ptr, lda, b_dmem_ptr, ldb, *beta, c_dmem_ptr, ldc, cuda_stream);break;
	case CUMPSGEMM_TF32TC:   layout_selector<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, nvcuda::wmma::precision::tf32, mtk::wmma::tcec::without_ec>(op_A, op_B, m, n, k, *alpha, a_dmem_ptr, lda, b_dmem_ptr, ldb, *beta, c_dmem_ptr, ldc, cuda_stream);break;
	case CUMPSGEMM_TF32TCEC: layout_selector<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, nvcuda::wmma::precision::tf32, mtk::wmma::tcec::with_ec   >(op_A, op_B, m, n, k, *alpha, a_dmem_ptr, lda, b_dmem_ptr, ldb, *beta, c_dmem_ptr, ldc, cuda_stream);break;
	default:break;
	}

	return CUBLAS_STATUS_SUCCESS;
}


template <class T>
cublasStatus_t cumpsgemm::gemm_stridedBatch(
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
		cudaStream_t cuda_stream
		) {
	switch (compute_mode) {
	case CUMPSGEMM_FP16TC:   stridedBatch_layout_selector<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, half                         , mtk::wmma::tcec::without_ec>(op_A, op_B, m, n, k, *alpha, a_dmem_ptr, lda, stridea, b_dmem_ptr, ldb, strideb, *beta, c_dmem_ptr, ldc, stridec, batch_count, cuda_stream);break;
	case CUMPSGEMM_FP16TCEC: stridedBatch_layout_selector<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, half                         , mtk::wmma::tcec::with_ec   >(op_A, op_B, m, n, k, *alpha, a_dmem_ptr, lda, stridea, b_dmem_ptr, ldb, strideb, *beta, c_dmem_ptr, ldc, stridec, batch_count, cuda_stream);break;
	case CUMPSGEMM_TF32TC:   stridedBatch_layout_selector<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, nvcuda::wmma::precision::tf32, mtk::wmma::tcec::without_ec>(op_A, op_B, m, n, k, *alpha, a_dmem_ptr, lda, stridea, b_dmem_ptr, ldb, strideb, *beta, c_dmem_ptr, ldc, stridec, batch_count, cuda_stream);break;
	case CUMPSGEMM_TF32TCEC: stridedBatch_layout_selector<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, nvcuda::wmma::precision::tf32, mtk::wmma::tcec::with_ec   >(op_A, op_B, m, n, k, *alpha, a_dmem_ptr, lda, stridea, b_dmem_ptr, ldb, strideb, *beta, c_dmem_ptr, ldc, stridec, batch_count, cuda_stream);break;
	default:break;
	}

	return CUBLAS_STATUS_SUCCESS;
}

extern "C" {
cublasStatus_t cuMpSGEMM_sgemm(
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const uint64_t m,
		const uint64_t n,
		const uint64_t k,
		const float* alpha,
		const float* const a_dmem_ptr, const uint64_t lda,
		const float* const b_dmem_ptr, const uint64_t ldb,
		const float* beta,
		float* const c_dmem_ptr, const uint64_t ldc,
		const cuMpSGEMM_compute_mode_t compute_mode,
		cudaStream_t cuda_stream
		) {
	assert(op_A != CUBLAS_OP_C);
	assert(op_B != CUBLAS_OP_C);
	return cumpsgemm::gemm<float>(
			op_A, op_B,
			m, n, k,
			alpha,
			a_dmem_ptr, lda,
			b_dmem_ptr, ldb,
			beta,
			c_dmem_ptr, ldc,
			compute_mode,
			cuda_stream
			);
}

cublasStatus_t cuMpSGEMM_cgemm(
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const uint64_t m,
		const uint64_t n,
		const uint64_t k,
		const cuComplex* alpha,
		const cuComplex* const a_dmem_ptr, const uint64_t lda,
		const cuComplex* const b_dmem_ptr, const uint64_t ldb,
		const cuComplex* beta,
		cuComplex* const c_dmem_ptr, const uint64_t ldc,
		const cuMpSGEMM_compute_mode_t compute_mode,
		cudaStream_t cuda_stream
		) {
	return cumpsgemm::gemm<cuComplex>(
			op_A, op_B,
			m, n, k,
			alpha,
			a_dmem_ptr, lda,
			b_dmem_ptr, ldb,
			beta,
			c_dmem_ptr, ldc,
			compute_mode,
			cuda_stream
			);
}

cublasStatus_t cuMpSGEMM_sgemm_strided_batch(
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const uint64_t m,
		const uint64_t n,
		const uint64_t k,
		const float* alpha,
		const float* const a_dmem_ptr, const uint64_t lda, const uint64_t stridea,
		const float* const b_dmem_ptr, const uint64_t ldb, const uint64_t strideb,
		const float* beta,
		float* const c_dmem_ptr, const uint64_t ldc, const uint64_t stridec,
		const uint64_t batch_count,
		const cuMpSGEMM_compute_mode_t compute_mode,
		cudaStream_t cuda_stream
		) {
	assert(op_A != CUBLAS_OP_C);
	assert(op_B != CUBLAS_OP_C);
	return cumpsgemm::gemm_stridedBatch<float>(
			op_A, op_B,
			m, n, k,
			alpha,
			a_dmem_ptr, lda, stridea,
			b_dmem_ptr, ldb, strideb,
			beta,
			c_dmem_ptr, ldc, stridec,
			batch_count,
			compute_mode,
			cuda_stream
			);
}

cublasStatus_t cuMpSGEMM_cgemm_strided_batch(
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const uint64_t m,
		const uint64_t n,
		const uint64_t k,
		const cuComplex* alpha,
		const cuComplex* const a_dmem_ptr, const uint64_t lda, const uint64_t stridea,
		const cuComplex* const b_dmem_ptr, const uint64_t ldb, const uint64_t strideb,
		const cuComplex* beta,
		cuComplex* const c_dmem_ptr, const uint64_t ldc, const uint64_t stridec,
		const uint64_t batch_count,
		const cuMpSGEMM_compute_mode_t compute_mode,
		cudaStream_t cuda_stream
		) {
	return cumpsgemm::gemm_stridedBatch<cuComplex>(
			op_A, op_B,
			m, n, k,
			alpha,
			a_dmem_ptr, lda, stridea,
			b_dmem_ptr, ldb, strideb,
			beta,
			c_dmem_ptr, ldc, stridec,
			batch_count,
			compute_mode,
			cuda_stream
			);
}
}
