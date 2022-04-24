#include <cstdlib>
#include <string>
#include <cumpsgemm/cumpsgemm.h>

extern "C" cuMpSGEMM_compute_mode_t cuMpSGEMM_get_compute_mode (
		const char* const func_name,
		cublasHandle_t const cublas_handle,
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const unsigned m, const unsigned n, const unsigned k
		) {
	const char* env_name = "CUMPSGEMM_COMPUTE_MODE";
	const char* env_val = getenv(env_name);

	if (env_val != nullptr) {
		const std::string env_val_str = env_val;
		if (env_val_str == "FP16TCEC")
			return CUMPSGEMM_FP16TCEC;
		if (env_val_str == "TF32TCEC")
			return CUMPSGEMM_TF32TCEC;
		if (env_val_str == "FP16TC")
			return CUMPSGEMM_FP16TC;
		if (env_val_str == "TF32TC")
			return CUMPSGEMM_TF32TC;
	}

	return CUMPSGEMM_CUBLAS;
}
