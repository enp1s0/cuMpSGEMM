#include "utils.hpp"
#include <cstdlib>
#include <cumpsgemm/cumpsgemm.h>
#include <string>

extern "C" cuMpSGEMM_compute_mode_t cuMpSGEMM_get_compute_mode(
    const char *const func_name, cublasHandle_t const cublas_handle,
    const cublasOperation_t op_A, const cublasOperation_t op_B,
    const unsigned m, const unsigned n, const unsigned k) {
  const char *env_name = "CUMPSGEMM_COMPUTE_MODE";
  const char *env_val = getenv(env_name);

  if (m <= 1024 || n <= 1024 || k <= 1024) {
    return CUMPSGEMM_CUBLAS_SIMT;
  }

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
    if (env_val_str == "CUBLAS_TF32TC")
      return CUMPSGEMM_CUBLAS_TF32TC;
    if (env_val_str == "CUBLAS_FP16TC")
      return CUMPSGEMM_CUBLAS_FP16TC;
    if (env_val_str == "CUBLAS_SIMT")
      return CUMPSGEMM_CUBLAS_SIMT;
    if (env_val_str == "CUBLAS")
      return CUMPSGEMM_CUBLAS;
    if (env_val_str == "DRY_RUN")
      return CUMPSGEMM_DRY_RUN;
    if (env_val_str == "AUTO")
      return CUMPSGEMM_AUTO;
    if (env_val_str == "FP16TCEC_SCALING")
      return CUMPSGEMM_FP16TCEC_SCALING;
    // if (env_val_str == "FP32_SIMT")
    //	return CUMPSGEMM_FP32_SIMT;
  }

  cuMpSGEMM_error("Unknown " + std::string(env_name) + " = " +
                  std::string(env_val) + ". Ignored");

  return CUMPSGEMM_CUBLAS;
}
