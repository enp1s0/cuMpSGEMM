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
#include "dynamic_scaling.hpp"
#include "culip.hpp"

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

const std::string gemm_Mx2x2_env_name = "CUMPSGEMM_CUSTOM_GEMM_MX2X2";
bool is_gemm_Mx2x2_enabled() {
	if (global_internal_gemm_Mx2x2_enabled) {
		return true;
	}

	const auto env = getenv(gemm_Mx2x2_env_name.c_str());
	if (env == nullptr || std::string(env) == "0") {
		return false;
	}

	return true;
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
	case CUMPSGEMM_UNDEFINED:
		return "UNDEFINED";
	case CUMPSGEMM_FP16TCEC_SCALING:
		return "FP16TCEC_SCALING";
	default:
		break;
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
	cudaStream_t cuda_stream;
	cublasGetStream(cublas_handle, &cuda_stream);

	if (m == 0 || n == 0 || k == 0 || lda == 0 || ldb == 0 || ldc == 0) {
		return CUBLAS_STATUS_INVALID_VALUE;
	}

	cumpsgemm::CULiP::profile_result profile_result;
	const auto profiling_flag = cumpsgemm::CULiP::is_profiling_enabled();

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

	// -----------------------------------
	// gemm_Mx2x2
	// -----------------------------------
	if (((m & (m - 1)) == 0) && n == 2 && k == 2 &&
			is_gemm_Mx2x2_enabled()) {

		if (profiling_flag) {
			const std::string func_name = std::string(std::is_same<T, float>::value ? "s" : "c") + "gemm_Mx2x2";
			snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s%s-m%lu-n%lu-k%lu", func_name.c_str(), cumpsgemm::CULiP::get_cublasOperation_t_string(op_A), cumpsgemm::CULiP::get_cublasOperation_t_string(op_B), m, n, k);
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.start_timestamp);
		}

		mtk::cugemm::gemm_Mx2x2(
				op_A, op_B,
				m,
				*alpha,
				a_dmem_ptr, lda,
				b_dmem_ptr, ldb,
				*beta,
				c_dmem_ptr, ldc,
				cuda_stream
				);

		if (profiling_flag) {
			// Record end rimestamp
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.end_timestamp);

			// Print result
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::print_profile_result, (void*)&profile_result);
		}

		return CUBLAS_STATUS_SUCCESS;
	}

	// -----------------------------------
	// gemm_2xNx2
	// -----------------------------------
	if (((n & (n - 1)) == 0) && m == 2 && k == 2 &&
			is_gemm_Mx2x2_enabled()) {

		if (profiling_flag) {
			const std::string func_name = std::string(std::is_same<T, float>::value ? "s" : "c") + "gemm_2xNx2";
			snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s%s-m%lu-n%lu-k%lu", func_name.c_str(), cumpsgemm::CULiP::get_cublasOperation_t_string(op_A), cumpsgemm::CULiP::get_cublasOperation_t_string(op_B), m, n, k);
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.start_timestamp);
		}

		mtk::cugemm::gemm_2xNx2(
				op_A, op_B,
				n,
				*alpha,
				a_dmem_ptr, lda,
				b_dmem_ptr, ldb,
				*beta,
				c_dmem_ptr, ldc,
				cuda_stream
				);

		if (profiling_flag) {
			// Record end rimestamp
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.end_timestamp);

			// Print result
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::print_profile_result, (void*)&profile_result);
		}

		return CUBLAS_STATUS_SUCCESS;
	}

	if (compute_mode == CUMPSGEMM_CUBLAS || compute_mode == CUMPSGEMM_CUBLAS_FP16TC || compute_mode == CUMPSGEMM_CUBLAS_TF32TC || compute_mode == CUMPSGEMM_CUBLAS_SIMT) {
		// -----------------------------------
		// cuBLAS
		// -----------------------------------
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

		if (profiling_flag) {
			snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s%s-m%lu-n%lu-k%lu", func_name, cumpsgemm::CULiP::get_cublasOperation_t_string(op_A), cumpsgemm::CULiP::get_cublasOperation_t_string(op_B), m, n, k);
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.start_timestamp);
		}

		res = (*func_ptr)(cublas_handle, op_A, op_B, m, n, k, alpha, a_dmem_ptr, lda, b_dmem_ptr, ldb, beta, c_dmem_ptr, ldc);

		if (profiling_flag) {
			// Record end rimestamp
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.end_timestamp);

			// Print result
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::print_profile_result, (void*)&profile_result);
		}

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
		// -----------------------------------
		// cuMpSGEMM
		// -----------------------------------
		if (profiling_flag) {
			const std::string func_name = std::string(std::is_same<T, float>::value ? "s" : "c") + "gemm_" + std::string(cuMpSGEMM_get_compute_mode_string(compute_mode));
			snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s%s-m%lu-n%lu-k%lu", func_name.c_str(), cumpsgemm::CULiP::get_cublasOperation_t_string(op_A), cumpsgemm::CULiP::get_cublasOperation_t_string(op_B), m, n, k);
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.start_timestamp);
		}

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

		if (profiling_flag) {
			// Record end rimestamp
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.end_timestamp);

			// Print result
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::print_profile_result, (void*)&profile_result);
		}
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
	cudaStream_t cuda_stream;
	cublasGetStream(cublas_handle, &cuda_stream);

	if (m == 0 || n == 0 || k == 0 || lda == 0 || ldb == 0 || ldc == 0 || batch_count == 0) {
		return CUBLAS_STATUS_INVALID_VALUE;
	}

	cumpsgemm::CULiP::profile_result profile_result;
	const auto profiling_flag = cumpsgemm::CULiP::is_profiling_enabled();

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

	// -----------------------------------
	// gemm_Mx2x2
	// -----------------------------------
	if (((m & (m - 1)) == 0) && n == 2 && k == 2 &&
			is_gemm_Mx2x2_enabled()) {

		if (profiling_flag) {
			const std::string func_name = std::string(std::is_same<T, float>::value ? "s" : "c") + "gemm_strided_batch_Mx2x2";
			snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s%s-m%lu-n%lu-k%lu-batchCount%lu",
					func_name.c_str(), cumpsgemm::CULiP::get_cublasOperation_t_string(op_A), cumpsgemm::CULiP::get_cublasOperation_t_string(op_B), m, n, k, batch_count);
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.start_timestamp);
		}

		mtk::cugemm::gemm_strided_batch_Mx2x2(
				op_A, op_B,
				m,
				*alpha,
				a_dmem_ptr, lda, stridea,
				b_dmem_ptr, ldb, strideb,
				*beta,
				c_dmem_ptr, ldc, stridec,
				batch_count,
				cuda_stream
				);

		if (profiling_flag) {
			// Record end rimestamp
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.end_timestamp);

			// Print result
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::print_profile_result, (void*)&profile_result);
		}

		return CUBLAS_STATUS_SUCCESS;
	}

	// -----------------------------------
	// gemm_2xNx2
	// -----------------------------------
	if (((n & (n - 1)) == 0) && m == 2 && k == 2 &&
			is_gemm_Mx2x2_enabled()) {

		if (profiling_flag) {
			const std::string func_name = std::string(std::is_same<T, float>::value ? "s" : "c") + "gemm_strided_batch_2xNx2";
			snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s%s-m%lu-n%lu-k%lu-batchCount%lu",
					func_name.c_str(), cumpsgemm::CULiP::get_cublasOperation_t_string(op_A), cumpsgemm::CULiP::get_cublasOperation_t_string(op_B), m, n, k, batch_count);
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.start_timestamp);
		}

		mtk::cugemm::gemm_strided_batch_2xNx2(
				op_A, op_B,
				n,
				*alpha,
				a_dmem_ptr, lda, stridea,
				b_dmem_ptr, ldb, strideb,
				*beta,
				c_dmem_ptr, ldc, stridec,
				batch_count,
				cuda_stream
				);

		if (profiling_flag) {
			// Record end rimestamp
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.end_timestamp);

			// Print result
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::print_profile_result, (void*)&profile_result);
		}

		return CUBLAS_STATUS_SUCCESS;
	}

	if (compute_mode == CUMPSGEMM_CUBLAS || compute_mode == CUMPSGEMM_CUBLAS_FP16TC || compute_mode == CUMPSGEMM_CUBLAS_TF32TC || compute_mode == CUMPSGEMM_CUBLAS_SIMT) {
		// -----------------------------------
		// cuBLAS
		// -----------------------------------
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

		if (profiling_flag) {
			snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s%s-m%lu-n%lu-k%lu-batchCount%lu", func_name, cumpsgemm::CULiP::get_cublasOperation_t_string(op_A), cumpsgemm::CULiP::get_cublasOperation_t_string(op_B), m, n, k, batch_count);
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.start_timestamp);
		}

		res = (*func_ptr)(cublas_handle, op_A, op_B, m, n, k, alpha, a_dmem_ptr, lda, stridea, b_dmem_ptr, ldb, strideb, beta, c_dmem_ptr, ldc, stridec, batch_count);

		if (profiling_flag) {
			// Record end rimestamp
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.end_timestamp);

			// Print result
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::print_profile_result, (void*)&profile_result);
		}

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
		// -----------------------------------
		// cuMpSGEMM
		// -----------------------------------
		if (profiling_flag) {
			const std::string func_name = std::string(std::is_same<T, float>::value ? "s" : "c") + "gemm_stridedBatch_" + std::string(cuMpSGEMM_get_compute_mode_string(compute_mode));
			snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s%s-m%lu-n%lu-k%lu-batchCount%lu", func_name.c_str(), cumpsgemm::CULiP::get_cublasOperation_t_string(op_A), cumpsgemm::CULiP::get_cublasOperation_t_string(op_B), m, n, k, batch_count);
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.start_timestamp);
		}

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

		if (profiling_flag) {
			// Record end rimestamp
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.end_timestamp);

			// Print result
			cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::print_profile_result, (void*)&profile_result);
		}
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

	cudaStream_t cuda_stream;
	cublasGetStream(handle, &cuda_stream);

	cumpsgemm::CULiP::profile_result profile_result;
	const auto profiling_flag = cumpsgemm::CULiP::is_profiling_enabled();

	cublasStatus_t (*func_ptr)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType_t, int, const void*, cudaDataType_t, int, const void*, void*, cudaDataType_t, int, cublasComputeType_t, cublasGemmAlgo_t);
	*(void**)(&func_ptr) = cuMpSGEMM_get_function_pointer(
			cublas_lib_name.c_str(),
			__func__
			);

	if (profiling_flag) {
		snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s%s-m%d-n%d-k%d", __func__, cumpsgemm::CULiP::get_cublasOperation_t_string(transa), cumpsgemm::CULiP::get_cublasOperation_t_string(transb), m, n, k);
		cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.start_timestamp);
	}

	const auto res = (*func_ptr)(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo);

	if (profiling_flag) {
		// Record end rimestamp
		cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.end_timestamp);

		// Print result
		cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::print_profile_result, (void*)&profile_result);
	}

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

	cudaStream_t cuda_stream;
	cublasGetStream(handle, &cuda_stream);

	cumpsgemm::CULiP::profile_result profile_result;
	const auto profiling_flag = cumpsgemm::CULiP::is_profiling_enabled();

	cublasStatus_t (*func_ptr)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void*, const void*, cudaDataType_t, int, long long int, const void*, cudaDataType_t, int, long long int, const void*, void*, cudaDataType_t, int, long long int, int, cublasComputeType_t, cublasGemmAlgo_t);
	*(void**)(&func_ptr) = cuMpSGEMM_get_function_pointer(
			cublas_lib_name.c_str(),
			__func__
			);

	if (profiling_flag) {
		snprintf(profile_result.function_name, profile_result.function_name_length - 1, "%s-%s%s-m%d-n%d-k%d-batch_count%d", __func__, cumpsgemm::CULiP::get_cublasOperation_t_string(transa), cumpsgemm::CULiP::get_cublasOperation_t_string(transb), m, n, k, batch_count);
		cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.start_timestamp);
	}

	const auto res = (*func_ptr)(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batch_count, computeType, algo);

	if (profiling_flag) {
		// Record end rimestamp
		cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::record_timestamp, (void*)&profile_result.end_timestamp);

		// Print result
		cumpsgemm::CULiP::launch_function(cuda_stream, &cumpsgemm::CULiP::print_profile_result, (void*)&profile_result);
	}

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
		const float underflow_threshold,
		const float underflow_tolerance_rate
		) {
	cumpsgemm::set_exp_stats_params(get_internal_global_handle(), ignore_threshold, underflow_threshold, underflow_tolerance_rate);
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

void cumpsgemm::hijack_control::set_dynamic_launch_buffer_by_exp_stats(
		const unsigned dynamic_launch_flag_buffer_id,
		const unsigned exp_stats_buffer_id_A,
		const unsigned exp_stats_buffer_id_B
		) {
	cumpsgemm::dynamic_scaling::set_dynamic_launch_buffer_by_exp_stats(
			get_internal_global_handle(),
			dynamic_launch_flag_buffer_id,
			exp_stats_buffer_id_A,
			exp_stats_buffer_id_B
			);
}

void cumpsgemm::hijack_control::scale_A(
		const unsigned exp_stats_buffer_id,
		const unsigned dynamic_launch_flag_buffer_id,
		const unsigned m,
		const unsigned n,
		float* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride
		) {
	cumpsgemm::dynamic_scaling::scale_A(
			cumpsgemm::hijack_control::get_internal_global_handle(),
			m, n,
			ptr, ld,
			batch_size,
			stride,
			exp_stats_buffer_id,
			dynamic_launch_flag_buffer_id
			);
}

void cumpsgemm::hijack_control::scale_B(
		const unsigned exp_stats_buffer_id,
		const unsigned dynamic_launch_flag_buffer_id,
		const unsigned m,
		const unsigned n,
		float* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride
		) {
	cumpsgemm::dynamic_scaling::scale_B(
			cumpsgemm::hijack_control::get_internal_global_handle(),
			m, n,
			ptr, ld,
			batch_size,
			stride,
			exp_stats_buffer_id,
			dynamic_launch_flag_buffer_id
			);
}

void cumpsgemm::hijack_control::scale_C(
		const unsigned exp_stats_buffer_A_id,
		const unsigned exp_stats_buffer_B_id,
		const unsigned dynamic_launch_flag_buffer_id,
		const unsigned m,
		const unsigned n,
		float* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride
		) {
	cumpsgemm::dynamic_scaling::scale_C(
			cumpsgemm::hijack_control::get_internal_global_handle(),
			m, n,
			ptr, ld,
			batch_size,
			stride,
			exp_stats_buffer_A_id,
			exp_stats_buffer_B_id,
			dynamic_launch_flag_buffer_id
			);
}

float cumpsgemm::hijack_control::get_max_exp(
		const unsigned dynamic_launch_flag_buffer_id
		) {
	return cumpsgemm::dynamic_scaling::get_max_exp(
			cumpsgemm::hijack_control::get_internal_global_handle(),
			dynamic_launch_flag_buffer_id
			);
}
void cumpsgemm::hijack_control::enable_custom_gemm_Mx2x2() {
	global_internal_gemm_Mx2x2_enabled  = true;
}

void cumpsgemm::hijack_control::disable_custom_gemm_Mx2x2() {
	global_internal_gemm_Mx2x2_enabled  = false;
}
