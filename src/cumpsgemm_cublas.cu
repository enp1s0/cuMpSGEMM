#include <string>
#include <cublas.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cumpsgemm/cumpsgemm.hpp>

namespace {
const std::string info_env_name = "CUMPSGEMM_INFO";
void cuMpSGEMM_log(
		const std::string str
		) {
	const auto env = getenv(info_env_name.c_str());
	if (env != nullptr && std::string(env) != "0") {
		std::fprintf(stdout, "[cuMpSGEMM LOG] %s\n",
				str.c_str());
	}
}

const std::string error_env_name = "CUMPSGEMM_ERROR";
void cuMpSGEMM_error(
		const std::string str
		) {
	const auto env = getenv(error_env_name.c_str());
	if (env != nullptr && std::string(env) != "0") {
		std::fprintf(stdout, "[cuMpSGEMM ERROR] %s\n",
				str.c_str());
	}
}

void* cuMpSGEMM_get_function_pointer(const std::string library_name, const std::string function_name) {

	// Open the library
	const auto lib_ptr = dlopen(library_name.c_str(), RTLD_NOW);
	if (lib_ptr == nullptr) {
		cuMpSGEMM_error("Failed to load " + library_name + ". Default rule will be used.");
		return nullptr;
	}

	// Get function pointer
	void* function_ptr = dlsym(lib_ptr, function_name.c_str());
	if (function_ptr == NULL) {
		cuMpSGEMM_error("Failed to load a function " + function_name + " during selecting hijacking function. Default rule will be used.");
		return nullptr;
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

extern "C" cuMpSGEMM_compute_mode_t cuMpSGEMM_get_compute_mode_internal (
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
		return cuMpSGEMM_get_compute_mode(func_name, cublas_handle, op_A, op_B, m, n, k);
	}

	return func(func_name, cublas_handle, op_A, op_B, m, n, k);
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
		cuMpSGEMM_get_compute_mode_internal(
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

template <class T>
cublasStatus_t cuMpSGEMM_stridedBatched_hijack_core(
		const char* const func_name,
		cublasHandle_t const cublas_handle,
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
		const uint64_t batch_count
		) {
	if (std::is_same<T, float>::value && (op_A == CUBLAS_OP_C || op_B == CUBLAS_OP_C)) {
		return CUBLAS_STATUS_INVALID_VALUE;
	}

	cudaStream_t cuda_stream;
	cublasGetStream(cublas_handle, &cuda_stream);

	cuMpSGEMM_compute_mode_t compute_mode =
		cuMpSGEMM_get_compute_mode_internal(
				func_name,
				cublas_handle,
				op_A,
				op_B,
				m, n, k
				);

	cuMpSGEMM_log(std::string(func_name) + " op=(" + get_cublas_op_str(op_A) + ", " + get_cublas_op_str(op_B) +
			"), shape=(" + std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + "), mode=" + cuMpSGEMM_get_compute_mode_string(compute_mode));

	if (compute_mode == CUMPSGEMM_CUBLAS) {
		cublasStatus_t (*func_ptr)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const T*, const T*, int, long long int, const T*, int, long long int, const T*, T*, int, long long int, int);
		*(void**)(&func_ptr) = cuMpSGEMM_get_function_pointer(
				cublas_lib_name.c_str(),
				func_name
				);
		return (*func_ptr)(cublas_handle, op_A, op_B, m, n, k, alpha, a_dmem_ptr, ldb, stridea, b_dmem_ptr, ldb, strideb, beta, c_dmem_ptr, ldc, stridec, batch_count);
	}

	return cumpsgemm::gemm_stridedBatch<T>(
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

cublasStatus_t cublasSgemmStridedBatched (
		cublasHandle_t cublas_handle,
		cublasOperation_t op_A,
		cublasOperation_t op_B,
		int m,
		int n,
		int k,
		const float* alpha,
		const float* a_dmem_ptr, int lda, long long int stridea,
		const float* b_dmem_ptr, int ldb, long long int strideb,
		const float* beta,
		float* c_dmem_ptr, int ldc, long long int stridec,
		const int batch_count
		) {
	cudaStream_t cuda_stream;
	cublasGetStream(cublas_handle, &cuda_stream);

	return cuMpSGEMM_stridedBatched_hijack_core<float>(
			__func__,
			cublas_handle,
			op_A, op_B,
			m, n, k,
			alpha,
			a_dmem_ptr, lda, stridea,
			b_dmem_ptr, ldb, strideb,
			beta,
			c_dmem_ptr, ldc, stridec,
			batch_count
			);
}

cublasStatus_t cublasCgemmStridedBatched (
		cublasHandle_t cublas_handle,
		cublasOperation_t op_A,
		cublasOperation_t op_B,
		int m,
		int n,
		int k,
		const cuComplex* alpha,
		const cuComplex* a_dmem_ptr, int lda, const long long int stridea,
		const cuComplex* b_dmem_ptr, int ldb, const long long int strideb,
		const cuComplex* beta,
		cuComplex* c_dmem_ptr, int ldc, const long long int stridec,
		const int batch_count
		) {
	cudaStream_t cuda_stream;
	cublasGetStream(cublas_handle, &cuda_stream);

	return cuMpSGEMM_stridedBatched_hijack_core<cuComplex>(
			__func__,
			cublas_handle,
			op_A, op_B,
			m, n, k,
			alpha,
			a_dmem_ptr, lda, stridea,
			b_dmem_ptr, ldb, strideb,
			beta,
			c_dmem_ptr, ldc, stridec,
			batch_count
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
		cuMpSGEMM_get_compute_mode_internal(
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

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const void *alpha, const void *A,
                            cudaDataType_t Atype, int lda, long long int strideA, const void *B,
                            cudaDataType_t Btype, int ldb, long long int strideB, const void *beta,
														void *C, cudaDataType_t Ctype, int ldc, long long int strideC,
														int batch_count,
														cublasComputeType_t computeType,
														cublasGemmAlgo_t algo) {
	cuMpSGEMM_compute_mode_t compute_mode =
		cuMpSGEMM_get_compute_mode_internal(
				__func__,
				handle,
				transa,
				transb,
				m, n, k
				);

	if (compute_mode != CUMPSGEMM_CUBLAS) {
		if (Atype == CUDA_R_32F && Btype == CUDA_R_32F && Ctype == CUDA_R_32F) {
			return cublasSgemmStridedBatched(
					handle,
					transa, transb,
					m, n, k,
					reinterpret_cast<const float*>(alpha),
					reinterpret_cast<const float*>(A), lda, strideA,
					reinterpret_cast<const float*>(B), ldb, strideB,
					reinterpret_cast<const float*>(beta),
					reinterpret_cast<float*>(C), ldc, strideC,
					batch_count
					);
		}
		if (Atype == CUDA_C_32F && Btype == CUDA_C_32F && Ctype == CUDA_C_32F) {
			return cublasCgemmStridedBatched(
					handle,
					transa, transb,
					m, n, k,
					reinterpret_cast<const cuComplex*>(alpha),
					reinterpret_cast<const cuComplex*>(A), lda, strideA,
					reinterpret_cast<const cuComplex*>(B), ldb, strideB,
					reinterpret_cast<const cuComplex*>(beta),
					reinterpret_cast<cuComplex*>(C), ldc, strideC,
					batch_count
					);
		}
	}
	cublasStatus_t (*func_ptr)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType_t, int, long long int, const void*, cudaDataType_t, int, long long int, const void*, void*, cudaDataType_t, int, long long int, int, cublasComputeType_t, cublasGemmAlgo_t);
	*(void**)(&func_ptr) = cuMpSGEMM_get_function_pointer(
			cublas_lib_name.c_str(),
			__func__
			);
	return (*func_ptr)(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batch_count, computeType, algo);
}
} // extern "C"
