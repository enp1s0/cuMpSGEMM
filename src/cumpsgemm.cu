#include <iostream>
#include <unistd.h>
#include <cassert>
#include <dlfcn.h>
#include <cublas.h>
#include <cumpsgemm/cumpsgemm.hpp>

#include "cumpsgemm_internal.hpp"

namespace {
const std::string debug_env_name = "CUMPSGEMM_DEBUG";
void cuMpSGEMM_log(
		const std::string str
		) {
	const auto env = getenv(debug_env_name.c_str());
	if (env != nullptr && std::string(env) == "1") {
		std::fprintf(stdout, "[cuMpSGEMM LOG] %s\n",
				str.c_str());
	}
}

void* cuMpSGEMM_get_function_pointer(const std::string library_name, const std::string function_name) {

	// Open the library
	const auto lib_ptr = dlopen(library_name.c_str(), RTLD_NOW);
	if (lib_ptr == nullptr) {
		cuMpSGEMM_log("Could not find the library " + library_name);
		return nullptr;
	}

	// Get function pointer
	void* function_ptr = dlsym(lib_ptr, function_name.c_str());
	if (function_ptr == NULL) {
		fprintf(stderr, "[cuMpSGEMM ERROR] Failed to load the function %s\n", __func__);
		exit(1);
	}

	return function_ptr;
}

std::string get_cublas_op_str(const cublasOperation_t op) {
	switch (op) {
	case CUBLAS_OP_C:
		return "C";
	case CUBLAS_OP_N:
		return "N";
	case CUBLAS_OP_T:
		return "T";
	default:
		return "?";
	}
}

const std::string rule_lib_name = "libcumpsgemm_rule.so";
const std::string cublas_lib_name = "libcublas.so";
} // noname namespace

extern "C" const char* cuMpSGEMM_get_compute_mode_string (
		const cuMpSGEMM_compute_mode_t mode
		) {
	switch (mode) {
	case CUMPSGEMM_CUBLAS:
		return "CUBLAS";
	case CUMPSGEMM_FP16TC:
		return "FP16TC";
	case CUMPSGEMM_FP16TCEC:
		return "FP16TCEC";
	case CUMPSGEMM_TF32TC:
		return "TF32TC";
	case CUMPSGEMM_TF32TCEC:
		return "TF32TCEC";
	}
	return "Unknown";
}

extern "C" cuMpSGEMM_compute_mode_t cuMpSGEMM_get_compute_mode (
		const char* const func_name,
		cublasHandle_t const cublas_handle,
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const unsigned m, const unsigned n, const unsigned k
		) {
	cuMpSGEMM_compute_mode_t (*func)(
			const char* const func_name,
			cublasHandle_t const cublas_handle,
			const cublasOperation_t op_A,
			const cublasOperation_t op_B,
			const unsigned m, const unsigned n, const unsigned k
			);
	*(void**)(&func) = cuMpSGEMM_get_function_pointer(rule_lib_name, __func__);

	if (func == nullptr) {
		return CUMPSGEMM_CUBLAS;
	}

	return func(func_name, cublas_handle, op_A, op_B, m, n, k);
}


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


extern "C" cublasStatus_t cuMpSGEMM_sgemm(
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

extern "C" cublasStatus_t cuMpSGEMM_cgemm(
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

template <class T>
cublasStatus_t cuMpSGEMM_hijack_core(
		const char* const func_name,
		cublasHandle_t const cublas_handle,
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const uint64_t m,
		const uint64_t n,
		const uint64_t k,
		const T* alpha,
		const T* const a_dmem_ptr, const uint64_t lda,
		const T* const b_dmem_ptr, const uint64_t ldb,
		const T* beta,
		T* const c_dmem_ptr, const uint64_t ldc
		) {
	if (std::is_same<T, float>::value && (op_A == CUBLAS_OP_C || op_B == CUBLAS_OP_C)) {
		return CUBLAS_STATUS_INVALID_VALUE;
	}

	cudaStream_t cuda_stream;
	cublasGetStream(cublas_handle, &cuda_stream);

	cuMpSGEMM_compute_mode_t compute_mode =
		cuMpSGEMM_get_compute_mode(
				func_name,
				cublas_handle,
				op_A,
				op_B,
				m, n, k
				);

	cuMpSGEMM_log(std::string(func_name) + " op=(" + get_cublas_op_str(op_A) + ", " + get_cublas_op_str(op_B) +
			"), shape=(" + std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + "), mode=" + cuMpSGEMM_get_compute_mode_string(compute_mode));

	if (compute_mode == CUMPSGEMM_CUBLAS) {
		cublasStatus_t (*func_ptr)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const T*, const T*, int, const T*, int, const T*, T*, int);
		*(void**)(&func_ptr) = cuMpSGEMM_get_function_pointer(
				cublas_lib_name.c_str(),
				func_name
				);
		return (*func_ptr)(cublas_handle, op_A, op_B, m, n, k, alpha, a_dmem_ptr, ldb, b_dmem_ptr, ldb, beta, c_dmem_ptr, ldc);
	}

	return cumpsgemm::gemm<T>(
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

// cuBLAS functions
extern "C" {
cublasStatus_t cublasSgemm (
		cublasHandle_t cublas_handle,
		cublasOperation_t op_A,
		cublasOperation_t op_B,
		int m,
		int n,
		int k,
		const float* alpha,
		const float* a_dmem_ptr, int lda,
		const float* b_dmem_ptr, int ldb,
		const float* beta,
		float* c_dmem_ptr, int ldc
		) {
	cudaStream_t cuda_stream;
	cublasGetStream(cublas_handle, &cuda_stream);

	return cuMpSGEMM_hijack_core<float>(
			__func__,
			cublas_handle,
			op_A, op_B,
			m, n, k,
			alpha,
			a_dmem_ptr, lda,
			b_dmem_ptr, ldb,
			beta,
			c_dmem_ptr, ldc
			);
}

cublasStatus_t cublasCgemm (
		cublasHandle_t cublas_handle,
		cublasOperation_t op_A,
		cublasOperation_t op_B,
		int m,
		int n,
		int k,
		const cuComplex* alpha,
		const cuComplex* a_dmem_ptr, int lda,
		const cuComplex* b_dmem_ptr, int ldb,
		const cuComplex* beta,
		cuComplex* c_dmem_ptr, int ldc
		) {
	cudaStream_t cuda_stream;
	cublasGetStream(cublas_handle, &cuda_stream);

	return cuMpSGEMM_hijack_core<cuComplex>(
			__func__,
			cublas_handle,
			op_A, op_B,
			m, n, k,
			alpha,
			a_dmem_ptr, lda,
			b_dmem_ptr, ldb,
			beta,
			c_dmem_ptr, ldc
			);
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const void *alpha, const void *A,
                            cudaDataType_t Atype, int lda, const void *B,
                            cudaDataType_t Btype, int ldb, const void *beta,
														void *C, cudaDataType_t Ctype, int ldc,
														cublasComputeType_t computeType,
														cublasGemmAlgo_t algo) {
	cuMpSGEMM_compute_mode_t compute_mode =
		cuMpSGEMM_get_compute_mode(
				__func__,
				handle,
				transa,
				transb,
				m, n, k
				);

	if (compute_mode != CUMPSGEMM_CUBLAS) {
		if (Atype == CUDA_R_32F && Btype == CUDA_R_32F && Ctype == CUDA_R_32F) {
			return cublasSgemm(
					handle,
					transa, transb,
					m, n, k,
					reinterpret_cast<const float*>(alpha),
					reinterpret_cast<const float*>(A), lda,
					reinterpret_cast<const float*>(B), ldb,
					reinterpret_cast<const float*>(beta),
					reinterpret_cast<float*>(C), ldc
					);
		}
		if (Atype == CUDA_C_32F && Btype == CUDA_C_32F && Ctype == CUDA_C_32F) {
			return cublasCgemm(
					handle,
					transa, transb,
					m, n, k,
					reinterpret_cast<const cuComplex*>(alpha),
					reinterpret_cast<const cuComplex*>(A), lda,
					reinterpret_cast<const cuComplex*>(B), ldb,
					reinterpret_cast<const cuComplex*>(beta),
					reinterpret_cast<cuComplex*>(C), ldc
					);
		}
	}
	cublasStatus_t (*func_ptr)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType_t, int, const void*, cudaDataType_t, int, const void*, void*, cudaDataType_t, int, cublasComputeType_t, cublasGemmAlgo_t);
	*(void**)(&func_ptr) = cuMpSGEMM_get_function_pointer(
			cublas_lib_name.c_str(),
			__func__
			);
	return (*func_ptr)(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
}
} // extern "C"
