#include <iostream>
#include <unistd.h>
#include <dlfcn.h>
#include <cumpsgemm/cumpsgemm.h>

const char* cuMpSGEMM_get_compute_mode_string (
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

namespace {
void* cuMpSGEMM_get_function_pointer(const std::string library_name, const std::string function_name) {

	// Open the library
	const auto lib_ptr = dlopen(library_name.c_str(), RTLD_NOW);
	if (lib_ptr == NULL) {
		std::fprintf(stderr, "[cuMpSGEMM ERROR] Failed to load the real library %s\n", library_name.c_str());
		exit(1);
	}

	// Get function pointer
	void* function_ptr = dlsym(lib_ptr, function_name.c_str());
	if (function_ptr == NULL) {
		fprintf(stderr, "[cuMpSGEMM ERROR] Failed to load the function %s\n", __func__);
		exit(1);
	}

	return function_ptr;
}

const std::string rule_lib_name = "libcumpsgemm_rule.so";
} // noname namespace

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

	return func(func_name, cublas_handle, op_A, op_B, m, n, k);
}
