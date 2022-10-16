#include <string>
#include <cublas.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cumpsgemm/cumpsgemm.hpp>
#include <cumpsgemm/hijack_control.hpp>
#include <cugemm_Mx2x2.hpp>
#include "handle.hpp"
#include "exp_stats.hpp"
#include "dynamic_launch.hpp"

namespace {

cuMpSGEMM_handle_t internal_global_cuMpSGEMM_handle = nullptr;
std::string internal_global_last_called_function_str = "";
bool global_internal_gemm_Mx2x2_enabled = false;

enum hijack_control_t {
	static_mode,
	dynamic_mode
} hijack_mode = dynamic_mode;
cuMpSGEMM_compute_mode_t internal_global_compute_mode = CUMPSGEMM_CUBLAS;

const std::string info_env_name = "CUMPSGEMM_INFO";
void cuMpSGEMM_log(
		const std::string str
		) {
	const auto env = getenv(info_env_name.c_str());
	if (env != nullptr && std::string(env) != "0") {
		std::fprintf(stdout, "[cuMpSGEMM LOG] %s\n",
				str.c_str());
		std::fflush(stdout);
	}
}

const std::string error_env_name = "CUMPSGEMM_ERROR_LOG";
void cuMpSGEMM_error(
		const std::string str
		) {
	const auto env = getenv(error_env_name.c_str());
	if (env != nullptr && std::string(env) != "0") {
		std::fprintf(stdout, "[cuMpSGEMM ERROR] %s\n",
				str.c_str());
		std::fflush(stdout);
	}
}

void cuMpSGEMM_warning(
		const std::string str
		) {
	const auto env = getenv(error_env_name.c_str());
	if (env != nullptr && std::string(env) != "0") {
		std::fprintf(stdout, "[cuMpSGEMM WARNING] %s\n",
				str.c_str());
		std::fflush(stdout);
	}
}

void* cuMpSGEMM_get_function_pointer(const std::string library_name, const std::string function_name) {

	// Open the library
	const auto lib_ptr = dlopen(library_name.c_str(), RTLD_NOW);
	if (lib_ptr == nullptr) {
		cuMpSGEMM_warning("Failed to load " + library_name + ". Default rule will be used.");
		return nullptr;
	}

	// Get function pointer
	void* function_ptr = dlsym(lib_ptr, function_name.c_str());
	if (function_ptr == NULL) {
		cuMpSGEMM_warning("Failed to load a function " + function_name + " during selecting hijacking function. Default rule will be used.");
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

cuMpSGEMM_handle_t cuMpSGEMM_get_internal_global_handle() {
	if (internal_global_cuMpSGEMM_handle == nullptr) {
		cuMpSGEMM_create(&internal_global_cuMpSGEMM_handle);
	}
	return internal_global_cuMpSGEMM_handle;
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
	case CUMPSGEMM_CUBLAS_SIMT:
		return "CUBLAS_SIMT";
	case CUMPSGEMM_CUBLAS_FP16TC:
		return "CUBLAS_FP16TC";
	case CUMPSGEMM_CUBLAS_TF32TC:
		return "CUBLAS_TF32TC";
	case CUMPSGEMM_DRY_RUN:
		return "DRY_RUN";
	case CUMPSGEMM_AUTO:
		return "AUTO";
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
	if (hijack_mode == dynamic_mode) {
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
	return internal_global_compute_mode;
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
			"), shape=(" + std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + "), mode=" + cuMpSGEMM_get_compute_mode_string(compute_mode) +
			"[" + (hijack_mode == dynamic_mode ? "dynamic" : "static") + "][exp_stats:" + (cumpsgemm::hijack_control::get_internal_global_handle()->exp_stats_handle->enabled ? "1" : "0") + "]");
	cumpsgemm::hijack_control::set_last_called_function_str(
			std::string(func_name) + "," +
			get_cublas_op_str(op_A) + "," +
			get_cublas_op_str(op_B) + "," +
			std::to_string(m) + "," +
			std::to_string(n) + "," +
			std::to_string(k) + "," +
			"1," + // batch_size
			cuMpSGEMM_get_compute_mode_string(compute_mode)
			);

	if (compute_mode == CUMPSGEMM_DRY_RUN) {
		return CUBLAS_STATUS_SUCCESS;
	}

	cublasStatus_t res;

	if (compute_mode == CUMPSGEMM_CUBLAS &&
			((m & (m - 1)) == 0) && n == 2 && k == 2 &&
			global_internal_gemm_Mx2x2_enabled) {

		mtk::cugemm::gemm_Mx2x2(
				op_A, op_B,
				m,
				*alpha,
				a_dmem_ptr, lda,
				b_dmem_ptr, ldb,
				*beta,
				c_dmem_ptr, ldc
				);

		return CUBLAS_STATUS_SUCCESS;
	}

	if (compute_mode == CUMPSGEMM_CUBLAS || compute_mode == CUMPSGEMM_CUBLAS_FP16TC || compute_mode == CUMPSGEMM_CUBLAS_TF32TC || compute_mode == CUMPSGEMM_CUBLAS_SIMT) {
		cublasMath_t math_mode;
		cublasGetMathMode(cublas_handle, &math_mode);
		if (compute_mode == CUMPSGEMM_CUBLAS_TF32TC) {
			cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
		} else if (compute_mode == CUMPSGEMM_CUBLAS_FP16TC) {
			cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
		} else if (compute_mode == CUMPSGEMM_CUBLAS_SIMT) {
			cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
		}

		cublasStatus_t (*func_ptr)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const T*, const T*, int, const T*, int, const T*, T*, int);
		*(void**)(&func_ptr) = cuMpSGEMM_get_function_pointer(
				cublas_lib_name.c_str(),
				func_name
				);
		if (func_ptr == nullptr) {
			cuMpSGEMM_error(std::string("Could not load cuBLAS function \"") + func_name + "\"");
		}
		res = (*func_ptr)(cublas_handle, op_A, op_B, m, n, k, alpha, a_dmem_ptr, lda, b_dmem_ptr, ldb, beta, c_dmem_ptr, ldc);

		// restore math mode
		cublasSetMathMode(cublas_handle, math_mode);

		if (cumpsgemm::hijack_control::get_internal_global_handle()->exp_stats_handle->enabled) {
			cumpsgemm::exp_stats::exp_stats_ext(
					cumpsgemm::hijack_control::get_internal_global_handle(),
					m, n,
					c_dmem_ptr, ldc,
					1,
					0
					);
		}

	} else {
		res = cumpsgemm::gemm<T>(
				cuMpSGEMM_get_internal_global_handle(),
				op_A, op_B,
				m, n, k,
				alpha,
				a_dmem_ptr, lda,
				b_dmem_ptr, ldb,
				beta,
				c_dmem_ptr, ldc,
				compute_mode
				);
	}

	return res;
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
			"), shape=(" + std::to_string(m) + ", " + std::to_string(n) + ", " + std::to_string(k) + "), batch=" + std::to_string(batch_count) + ", mode=" + cuMpSGEMM_get_compute_mode_string(compute_mode) +
			"[" + (hijack_mode == dynamic_mode ? "dynamic" : "static") + "][exp_stats:" + (cumpsgemm::hijack_control::get_internal_global_handle()->exp_stats_handle->enabled ? "1" : "0") + "]");

	cumpsgemm::hijack_control::set_last_called_function_str(
			std::string(func_name) + "," +
			get_cublas_op_str(op_A) + "," +
			get_cublas_op_str(op_B) + "," +
			std::to_string(m) + "," +
			std::to_string(n) + "," +
			std::to_string(k) + "," +
			std::to_string(batch_count) + "," +
			cuMpSGEMM_get_compute_mode_string(compute_mode)
			);

	if (compute_mode == CUMPSGEMM_DRY_RUN) {
		return CUBLAS_STATUS_SUCCESS;
	}

	cublasStatus_t res;

	if (compute_mode == CUMPSGEMM_CUBLAS &&
			((m & (m - 1)) == 0) && n == 2 && k == 2 &&
			global_internal_gemm_Mx2x2_enabled) {

		mtk::cugemm::gemm_strided_batch_Mx2x2(
				op_A, op_B,
				m,
				*alpha,
				a_dmem_ptr, lda, stridea,
				b_dmem_ptr, ldb, strideb,
				*beta,
				c_dmem_ptr, ldc, stridec,
				batch_count
				);

		return CUBLAS_STATUS_SUCCESS;
	}

	if (compute_mode == CUMPSGEMM_CUBLAS || compute_mode == CUMPSGEMM_CUBLAS_FP16TC || compute_mode == CUMPSGEMM_CUBLAS_TF32TC || compute_mode == CUMPSGEMM_CUBLAS_SIMT) {
		cublasMath_t math_mode;
		cublasGetMathMode(cublas_handle, &math_mode);
		if (compute_mode == CUMPSGEMM_CUBLAS_TF32TC) {
			cublasSetMathMode(cublas_handle, CUBLAS_TF32_TENSOR_OP_MATH);
		} else if (compute_mode == CUMPSGEMM_CUBLAS_FP16TC) {
			cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
		} else if (compute_mode == CUMPSGEMM_CUBLAS_SIMT) {
			cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
		}

		cublasStatus_t (*func_ptr)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const T*, const T*, int, long long int, const T*, int, long long int, const T*, T*, int, long long int, int);
		*(void**)(&func_ptr) = cuMpSGEMM_get_function_pointer(
				cublas_lib_name.c_str(),
				func_name
				);
		if (func_ptr == nullptr) {
			cuMpSGEMM_error(std::string("Could not load cuBLAS function \"") + func_name + "\"");
		}
		res = (*func_ptr)(cublas_handle, op_A, op_B, m, n, k, alpha, a_dmem_ptr, lda, stridea, b_dmem_ptr, ldb, strideb, beta, c_dmem_ptr, ldc, stridec, batch_count);
		cublasSetMathMode(cublas_handle, math_mode);

		if (cumpsgemm::hijack_control::get_internal_global_handle()->exp_stats_handle->enabled) {
			cumpsgemm::exp_stats::exp_stats_ext(
					cumpsgemm::hijack_control::get_internal_global_handle(),
					m, n,
					c_dmem_ptr, ldc,
					batch_count,
					stridec
					);
		}
	} else {

		res = cumpsgemm::gemm_stridedBatch<T>(
				cuMpSGEMM_get_internal_global_handle(),
				op_A, op_B,
				m, n, k,
				alpha,
				a_dmem_ptr, lda, stridea,
				b_dmem_ptr, ldb, strideb,
				beta,
				c_dmem_ptr, ldc, stridec,
				batch_count,
				compute_mode
				);
	}
	return res;
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
	cublasStatus_t (*func_ptr)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType_t, int, const void*, cudaDataType_t, int, const void*, void*, cudaDataType_t, int, cublasComputeType_t, cublasGemmAlgo_t);
	*(void**)(&func_ptr) = cuMpSGEMM_get_function_pointer(
			cublas_lib_name.c_str(),
			__func__
			);
	const auto res = (*func_ptr)(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);
	return res;
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
	cublasStatus_t (*func_ptr)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType_t, int, long long int, const void*, cudaDataType_t, int, long long int, const void*, void*, cudaDataType_t, int, long long int, int, cublasComputeType_t, cublasGemmAlgo_t);
	*(void**)(&func_ptr) = cuMpSGEMM_get_function_pointer(
			cublas_lib_name.c_str(),
			__func__
			);
	const auto res = (*func_ptr)(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batch_count, computeType, algo);
	return res;
}
} // extern "C"

cuMpSGEMM_handle* cumpsgemm::hijack_control::get_internal_global_handle() {
	if (internal_global_cuMpSGEMM_handle == nullptr) {
		cuMpSGEMM_create(&internal_global_cuMpSGEMM_handle);
	}
	return internal_global_cuMpSGEMM_handle;
}

void cumpsgemm::hijack_control::set_compute_mode(const cuMpSGEMM_compute_mode_t mode) {
	internal_global_compute_mode = mode;
	hijack_mode = static_mode;
}

void cumpsgemm::hijack_control::unset_compute_mode() {
	hijack_mode = dynamic_mode;
}

std::pair<std::size_t, std::size_t> cumpsgemm::hijack_control::get_exp_stats(const unsigned buffer_id) {
	return cumpsgemm::exp_stats::get_exp_stats(get_internal_global_handle(), buffer_id);
}

unsigned cumpsgemm::hijack_control::get_current_exp_stats_buffer_id() {
	return cumpsgemm::exp_stats::get_current_exp_stats_buffer_id(cuMpSGEMM_get_internal_global_handle());
}

void cumpsgemm::hijack_control::enable_exp_stats() {
	cumpsgemm::enable_exp_stats(get_internal_global_handle());
}

void cumpsgemm::hijack_control::disable_exp_stats() {
	cumpsgemm::disable_exp_stats(get_internal_global_handle());
	cumpsgemm::hijack_control::unset_compute_mode();
}

void cumpsgemm::hijack_control::set_exp_stats_params(
		const float ignore_threshold,
		const float lost_threshold
		) {
	cumpsgemm::set_exp_stats_params(get_internal_global_handle(), ignore_threshold, lost_threshold);
}

bool cumpsgemm::hijack_control::is_exp_stats_enabled() {
	return get_internal_global_handle()->exp_stats_handle->enabled;
}

void cumpsgemm::hijack_control::reset_exp_stats_buffer_id() {
	cumpsgemm::exp_stats::reset_exp_stats_buffer_id(get_internal_global_handle());
}

void cumpsgemm::hijack_control::exp_stats(
		const unsigned m,
		const unsigned n,
		const float* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride
		) {
	cumpsgemm::exp_stats::exp_stats_ext(
			get_internal_global_handle(),
			m, n,
			ptr, ld,
			batch_size, stride
			);
}

void cumpsgemm::hijack_control::download_exp_stats_result(const unsigned id) {
	cumpsgemm::exp_stats::download_exp_stats(get_internal_global_handle(), id);
}

std::string cumpsgemm::hijack_control::get_last_called_function_str() {
	return internal_global_last_called_function_str;
}

void cumpsgemm::hijack_control::set_last_called_function_str(
		const std::string func_str
		) {
	internal_global_last_called_function_str = func_str;
}

void cumpsgemm::hijack_control::clear_last_called_function_str() {
	cumpsgemm::hijack_control::set_last_called_function_str("");
}

unsigned cumpsgemm::hijack_control::get_next_dynamic_launch_flag_buffer_id() {
	return cumpsgemm::dynamic_launch::get_next_dynamic_launch_flag_buffer_id(get_internal_global_handle());
}

void cumpsgemm::hijack_control::set_dynamic_launch_flag_buffer_id_use(unsigned id) {
	cumpsgemm::dynamic_launch::set_dynamic_launch_flag_buffer_id(get_internal_global_handle(), id);
}

namespace {
__global__ void dynamic_launch_flag_buffer_id_by_exp_stats_kernel(
		int* const flag_buffer_ptr,
		const cumpsgemm::counter_t* const total_counter_A_ptr,
		const cumpsgemm::counter_t* const lost_counter_A_ptr,
		const cumpsgemm::counter_t* const total_counter_B_ptr,
		const cumpsgemm::counter_t* const lost_counter_B_ptr,
		const float rate_threshold
		) {
	const auto pA = (static_cast<float>(*lost_counter_A_ptr) / *total_counter_A_ptr) < rate_threshold;
	const auto pB = (static_cast<float>(*lost_counter_B_ptr) / *total_counter_B_ptr) < rate_threshold;
	if (pA && pB) {
		*flag_buffer_ptr = CUMPSGEMM_FP16TCEC;
	} else {
		*flag_buffer_ptr = CUMPSGEMM_TF32TCEC;
	}
}
} // unnamed namespace

void cumpsgemm::hijack_control::set_dynamic_launch_flag_buffer_by_exp_stats(
		const unsigned exp_stats_buffer_id_A,
		const unsigned exp_stats_buffer_id_B,
		const unsigned dynamic_launch_flag_buffer_id,
		const float ratio_threshold
		) {
	const auto handle = get_internal_global_handle();
	const auto cuda_stream = handle->cuda_stream;

	dynamic_launch_flag_buffer_id_by_exp_stats_kernel<<<1, 1, 0, cuda_stream>>>(
			handle->dynamic_launch_handle->frag_buffer + dynamic_launch_flag_buffer_id,
			handle->exp_stats_handle->dev_total_counter_buffer + exp_stats_buffer_id_A,
			handle->exp_stats_handle->dev_lost_counter_buffer  + exp_stats_buffer_id_A,
			handle->exp_stats_handle->dev_total_counter_buffer + exp_stats_buffer_id_B,
			handle->exp_stats_handle->dev_lost_counter_buffer  + exp_stats_buffer_id_B,
			ratio_threshold
			);
}

void cumpsgemm::hijack_control::enable_custom_gemm_Mx2x2() {
	global_internal_gemm_Mx2x2_enabled  = true;
}

void cumpsgemm::hijack_control::disable_custom_gemm_Mx2x2() {
	global_internal_gemm_Mx2x2_enabled  = true;
}
