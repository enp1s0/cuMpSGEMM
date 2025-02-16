#include "culip.hpp"
#include "dynamic_launch.hpp"
#include "dynamic_launch_utils.hpp"
#include "dynamic_scaling.hpp"
#include "exp_stats.hpp"
#include "handle.hpp"
#include "utils.hpp"
#include <cugemm_Mx2x2.hpp>
#include <cumpsgemm/cumpsgemm.hpp>
#include <cumpsgemm/hijack_control.hpp>
#include <cutf/memory.hpp>
#include <dlfcn.h>
#include <iomanip>
#include <sstream>
#include <string>
#include <unistd.h>

#ifndef CUBLASAPI
#define CUBLASAPI
#endif

namespace {
std::string get_XeY_format_string(const double a) {
  std::stringstream ss;

  ss << std::scientific << a;

  return ss.str();
}
cuMpSGEMM_handle_t internal_global_cuMpSGEMM_handle = nullptr;
std::string internal_global_last_called_function_str = "";
bool global_internal_gemm_Mx2x2_enabled = false;
bool restore_AB = true;
cumpsgemm::hijack_control::control_function_t internal_global_control_func;

enum hijack_control_t { static_mode, dynamic_mode } hijack_mode = dynamic_mode;
cuMpSGEMM_compute_mode_t internal_global_compute_mode = CUMPSGEMM_CUBLAS;

void *cuMpSGEMM_get_function_pointer(const std::string function_name,
                                     const std::string library_name = "") {
  // Get function pointer
  void *function_ptr = nullptr;
  if (library_name != "") {
    // Open the library
    const auto lib_ptr = dlopen(library_name.c_str(), RTLD_NOW);
    if (lib_ptr == nullptr) {
      cuMpSGEMM_warning("Failed to load " + library_name +
                        ". Default rule will be used.");
      return nullptr;
    }

    function_ptr = dlsym(lib_ptr, function_name.c_str());
  } else {
    function_ptr = dlsym(RTLD_NEXT, function_name.c_str());
  }
  if (function_ptr == nullptr) {
    cuMpSGEMM_warning(
        "Failed to load a function " + function_name +
        " during selecting hijacking function. Default rule will be used.");
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

template <class T> inline cudaDataType_t get_cuda_data_type();
template <> inline cudaDataType_t get_cuda_data_type<float>() {
  return CUDA_R_32F;
}
template <> inline cudaDataType_t get_cuda_data_type<cuComplex>() {
  return CUDA_C_32F;
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

cuMpSGEMM_handle_t cuMpSGEMM_get_internal_global_handle() {
  if (internal_global_cuMpSGEMM_handle == nullptr) {
    cuMpSGEMM_log("Initialize cuMpSGEMM handle...");
    if (cuMpSGEMM_create(&internal_global_cuMpSGEMM_handle) !=
        CUBLAS_STATUS_SUCCESS) {
      cuMpSGEMM_error("Initialization failed.");
    }

    const auto init_float_by_env = [&](const std::string env_str,
                                       const float default_value) {
      const auto env = getenv(env_str.c_str());
      if (env != nullptr) {
        return std::stof(env);
      }
      return default_value;
    };

    const auto init_int_by_env = [&](const std::string env_str,
                                     const int default_value) {
      const auto env = getenv(env_str.c_str());
      if (env != nullptr) {
        return std::stoi(env);
      }
      return default_value;
    };

    // AUTO mode configure
    const auto ignore_threshold =
        init_float_by_env("CUMPSGEMM_AUTO_IGNORE_THRESHOLD", 0);
    const auto underflow_threshold =
        init_float_by_env("CUMPSGEMM_AUTO_UNDERFLOW_THRESHOLD", 1.f / 32768);
    const auto underflow_tolerance_rate =
        init_float_by_env("CUMPSGEMM_AUTO_UNDERFLOW_TOLERANCE_RATE", 0);
    const auto restore_AB_scaling =
        init_int_by_env("CUMPSGEMM_AUTO_RESTORE_AB_SCALING", 1);

    cuMpSGEMM_log("AUTO config: ignore_threshold=" +
                  get_XeY_format_string(ignore_threshold) + " @Init");
    cuMpSGEMM_log("AUTO config: underflow_threshold=" +
                  get_XeY_format_string(underflow_threshold) + " @Init");
    cuMpSGEMM_log("AUTO config: underflow_tolerance_rate=" +
                  get_XeY_format_string(underflow_tolerance_rate) + " @Init");
    cuMpSGEMM_log("AUTO config: restore_AB_scaling=" +
                  std::to_string(restore_AB_scaling) + " @Init");
    cuMpSGEMM_log(
        "CUSTOM_GEMM_MX2X2: " +
        std::string(is_gemm_Mx2x2_enabled() ? "enabled" : "disabled") +
        " @Init");

    cumpsgemm::set_exp_stats_params(cuMpSGEMM_get_internal_global_handle(),
                                    ignore_threshold, underflow_threshold,
                                    underflow_tolerance_rate);
    restore_AB = restore_AB_scaling;
  }

  return internal_global_cuMpSGEMM_handle;
}

const std::string rule_lib_name = "libcumpsgemm_rule.so";
const std::string cublas_lib_name = "libcublas.so";
} // namespace

extern "C" const char *
cuMpSGEMM_get_compute_mode_string(const cuMpSGEMM_compute_mode_t mode) {
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
  case CUMPSGEMM_FP32_SIMT:
    return "FP32_SIMT";
  default:
    break;
  }
  return "Unknown";
}

extern "C" cuMpSGEMM_compute_mode_t cuMpSGEMM_get_compute_mode_internal(
    const char *const func_name, cublasHandle_t const cublas_handle,
    const cublasOperation_t op_A, const cublasOperation_t op_B,
    const unsigned m, const unsigned n, const unsigned k) {
  if (hijack_mode == dynamic_mode) {
    if (internal_global_control_func) {
      return internal_global_control_func(op_A, op_B, m, n, k);
    }
    cuMpSGEMM_compute_mode_t (*func)(
        const char *const func_name, cublasHandle_t const cublas_handle,
        const cublasOperation_t op_A, const cublasOperation_t op_B,
        const unsigned m, const unsigned n, const unsigned k);
    *(void **)(&func) = cuMpSGEMM_get_function_pointer(__func__, rule_lib_name);

    if (func == nullptr) {
      return cuMpSGEMM_get_compute_mode(func_name, cublas_handle, op_A, op_B, m,
                                        n, k);
    }

    return func(func_name, cublas_handle, op_A, op_B, m, n, k);
  }
  return internal_global_compute_mode;
}

template <class T>
cublasStatus_t cuMpSGEMM_hijack_core(
    const char *const func_name, cublasHandle_t const cublas_handle,
    const cublasOperation_t op_A, const cublasOperation_t op_B,
    const uint64_t m, const uint64_t n, const uint64_t k, const T *alpha,
    const T *const a_dmem_ptr, const uint64_t lda, const T *const b_dmem_ptr,
    const uint64_t ldb, const T *beta, T *const c_dmem_ptr,
    const uint64_t ldc) {
  cudaStream_t cuda_stream;
  cublasGetStream(cublas_handle, &cuda_stream);

  if (m == 0 || n == 0 || k == 0 || lda == 0 || ldb == 0 || ldc == 0) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  cumpsgemm::CULiP::profile_result profile_result;
  const auto profiling_flag = cumpsgemm::CULiP::is_profiling_enabled();

  cuMpSGEMM_compute_mode_t compute_mode = cuMpSGEMM_get_compute_mode_internal(
      func_name, cublas_handle, op_A, op_B, m, n, k);

  cuMpSGEMM_log(
      std::string(func_name) + " op=(" + get_cublas_op_str(op_A) + ", " +
      get_cublas_op_str(op_B) + "), shape=(" + std::to_string(m) + ", " +
      std::to_string(n) + ", " + std::to_string(k) +
      "), mode=" + cuMpSGEMM_get_compute_mode_string(compute_mode) + "[" +
      (hijack_mode == dynamic_mode ? "dynamic" : "static") + "][exp_stats:" +
      (cumpsgemm::hijack_control::get_internal_global_handle()
               ->exp_stats_handle->enabled
           ? "1"
           : "0") +
      "]");
  cumpsgemm::hijack_control::set_last_called_function_str(
      std::string(func_name) + "," + get_cublas_op_str(op_A) + "," +
      get_cublas_op_str(op_B) + "," + std::to_string(m) + "," +
      std::to_string(n) + "," + std::to_string(k) + "," + "1," + // batch_size
      cuMpSGEMM_get_compute_mode_string(compute_mode));

  if (compute_mode == CUMPSGEMM_DRY_RUN) {
    return CUBLAS_STATUS_SUCCESS;
  }

  cublasStatus_t res;

  // -----------------------------------
  // gemm_Mx2x2
  // -----------------------------------
  if (((m & (m - 1)) == 0) && n == 2 && k == 2 && is_gemm_Mx2x2_enabled()) {

    if (profiling_flag) {
      const std::string func_name =
          std::string(std::is_same<T, float>::value ? "s" : "c") + "gemm_Mx2x2";
      snprintf(profile_result.function_name,
               profile_result.function_name_length - 1,
               "%s-%s%s-m%lu-n%lu-k%lu", func_name.c_str(),
               cumpsgemm::CULiP::get_cublasOperation_t_string(op_A),
               cumpsgemm::CULiP::get_cublasOperation_t_string(op_B), m, n, k);
      cumpsgemm::CULiP::launch_function(
          cuda_stream, &cumpsgemm::CULiP::record_timestamp,
          (void *)&profile_result.start_timestamp);
    }
    cuMpSGEMM_log(" +---> gemm_Mx2x2");

    mtk::cugemm::gemm_Mx2x2(op_A, op_B, m, *alpha, a_dmem_ptr, lda, b_dmem_ptr,
                            ldb, *beta, c_dmem_ptr, ldc, cuda_stream);

    if (profiling_flag) {
      // Record end rimestamp
      cumpsgemm::CULiP::launch_function(cuda_stream,
                                        &cumpsgemm::CULiP::record_timestamp,
                                        (void *)&profile_result.end_timestamp);

      // Print result
      cumpsgemm::CULiP::launch_function(cuda_stream,
                                        &cumpsgemm::CULiP::print_profile_result,
                                        (void *)&profile_result);
    }

    return CUBLAS_STATUS_SUCCESS;
  }

  // -----------------------------------
  // gemm_2xNx2
  // -----------------------------------
  if (((n & (n - 1)) == 0) && m == 2 && k == 2 && is_gemm_Mx2x2_enabled()) {

    if (profiling_flag) {
      const std::string func_name =
          std::string(std::is_same<T, float>::value ? "s" : "c") + "gemm_2xNx2";
      snprintf(profile_result.function_name,
               profile_result.function_name_length - 1,
               "%s-%s%s-m%lu-n%lu-k%lu", func_name.c_str(),
               cumpsgemm::CULiP::get_cublasOperation_t_string(op_A),
               cumpsgemm::CULiP::get_cublasOperation_t_string(op_B), m, n, k);
      cumpsgemm::CULiP::launch_function(
          cuda_stream, &cumpsgemm::CULiP::record_timestamp,
          (void *)&profile_result.start_timestamp);
    }
    cuMpSGEMM_log(" +---> gemm_2xNx2");

    mtk::cugemm::gemm_2xNx2(op_A, op_B, n, *alpha, a_dmem_ptr, lda, b_dmem_ptr,
                            ldb, *beta, c_dmem_ptr, ldc, cuda_stream);

    if (profiling_flag) {
      // Record end rimestamp
      cumpsgemm::CULiP::launch_function(cuda_stream,
                                        &cumpsgemm::CULiP::record_timestamp,
                                        (void *)&profile_result.end_timestamp);

      // Print result
      cumpsgemm::CULiP::launch_function(cuda_stream,
                                        &cumpsgemm::CULiP::print_profile_result,
                                        (void *)&profile_result);
    }

    return CUBLAS_STATUS_SUCCESS;
  }

  if (compute_mode == CUMPSGEMM_CUBLAS ||
      compute_mode == CUMPSGEMM_CUBLAS_FP16TC ||
      compute_mode == CUMPSGEMM_CUBLAS_TF32TC ||
      compute_mode == CUMPSGEMM_CUBLAS_SIMT) {
    // -----------------------------------
    // cuBLAS
    // -----------------------------------
    cublasGemmAlgo_t gemm_algo = CUBLAS_GEMM_DEFAULT;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    const cudaDataType_t io_datat_type = get_cuda_data_type<T>();
    if (compute_mode == CUMPSGEMM_CUBLAS_TF32TC) {
      gemm_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
      compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    } else if (compute_mode == CUMPSGEMM_CUBLAS_FP16TC) {
      gemm_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
      compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
    } else if (compute_mode == CUMPSGEMM_CUBLAS_SIMT) {
      // Do nothing
    } else {
      const std::string func_name_str(func_name);
      if (func_name_str == "cublasSgemm" || func_name_str == "cublasCgemm") {
        cublasMath_t math_mode;
        cublasGetMathMode(cublas_handle, &math_mode);
        switch (math_mode) {
        case CUBLAS_DEFAULT_MATH:
        case CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION:
          // Do nothing
          break;
        case CUBLAS_TF32_TENSOR_OP_MATH:
          gemm_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
          compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
          break;
        case CUBLAS_TENSOR_OP_MATH:
          gemm_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
          compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
          break;
        default:
          break;
        }
      }
    }

    cublasStatus_t (*func_ptr)(
        cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
        const void *, const void *, cudaDataType, int, const void *,
        cudaDataType, int, const void *, void *, cudaDataType, int,
        cublasComputeType_t, cublasGemmAlgo_t);
    *(void **)(&func_ptr) = cuMpSGEMM_get_function_pointer("cublasGemmEx");
    if (func_ptr == nullptr) {
      cuMpSGEMM_error(std::string("Could not load the cuBLAS function \"") +
                      func_name + "\"");
    }

    if (profiling_flag) {
      snprintf(profile_result.function_name,
               profile_result.function_name_length - 1,
               "%s-%s%s-m%lu-n%lu-k%lu", func_name,
               cumpsgemm::CULiP::get_cublasOperation_t_string(op_A),
               cumpsgemm::CULiP::get_cublasOperation_t_string(op_B), m, n, k);
      cumpsgemm::CULiP::launch_function(
          cuda_stream, &cumpsgemm::CULiP::record_timestamp,
          (void *)&profile_result.start_timestamp);
    }

    res = (*func_ptr)(cublas_handle, op_A, op_B, m, n, k, alpha, a_dmem_ptr,
                      io_datat_type, lda, b_dmem_ptr, io_datat_type, ldb, beta,
                      c_dmem_ptr, io_datat_type, ldc, compute_type, gemm_algo);

    if (profiling_flag) {
      // Record end rimestamp
      cumpsgemm::CULiP::launch_function(cuda_stream,
                                        &cumpsgemm::CULiP::record_timestamp,
                                        (void *)&profile_result.end_timestamp);

      // Print result
      cumpsgemm::CULiP::launch_function(cuda_stream,
                                        &cumpsgemm::CULiP::print_profile_result,
                                        (void *)&profile_result);
    }

    if (cumpsgemm::hijack_control::get_internal_global_handle()
            ->exp_stats_handle->enabled) {
      cumpsgemm::exp_stats::exp_stats_ext(
          cumpsgemm::hijack_control::get_internal_global_handle(), m, n,
          c_dmem_ptr, ldc, 1, 0);
    }

  } else {
    // -----------------------------------
    // cuMpSGEMM
    // -----------------------------------
    if (profiling_flag) {
      const std::string func_name =
          std::string(std::is_same<T, float>::value ? "s" : "c") + "gemm_" +
          std::string(cuMpSGEMM_get_compute_mode_string(compute_mode));
      snprintf(profile_result.function_name,
               profile_result.function_name_length - 1,
               "%s-%s%s-m%lu-n%lu-k%lu", func_name.c_str(),
               cumpsgemm::CULiP::get_cublasOperation_t_string(op_A),
               cumpsgemm::CULiP::get_cublasOperation_t_string(op_B), m, n, k);
      cumpsgemm::CULiP::launch_function(
          cuda_stream, &cumpsgemm::CULiP::record_timestamp,
          (void *)&profile_result.start_timestamp);
    }

    unsigned A_exp_stats_id, B_exp_stats_id, dynamic_launch_id;
    if (compute_mode == CUMPSGEMM_AUTO) {
      // Exp stats
      cumpsgemm::exp_stats::exp_stats_ext(
          cuMpSGEMM_get_internal_global_handle(), (op_A == CUBLAS_OP_N ? m : k),
          (op_A == CUBLAS_OP_N ? k : m), a_dmem_ptr, lda, 1, 0);
      A_exp_stats_id = cumpsgemm::exp_stats::get_current_exp_stats_buffer_id(
          cuMpSGEMM_get_internal_global_handle());
      cumpsgemm::exp_stats::exp_stats_ext(
          cuMpSGEMM_get_internal_global_handle(), (op_B == CUBLAS_OP_N ? k : n),
          (op_B == CUBLAS_OP_N ? n : k), b_dmem_ptr, ldb, 1, 0);
      B_exp_stats_id = cumpsgemm::exp_stats::get_current_exp_stats_buffer_id(
          cuMpSGEMM_get_internal_global_handle());

      // Kernel decision
      dynamic_launch_id =
          cumpsgemm::dynamic_launch::get_next_dynamic_launch_flag_buffer_id(
              cuMpSGEMM_get_internal_global_handle());
      cumpsgemm::dynamic_scaling::set_dynamic_launch_buffer_by_exp_stats(
          cuMpSGEMM_get_internal_global_handle(), dynamic_launch_id,
          A_exp_stats_id, B_exp_stats_id);

      cuMpSGEMM_run_if_env_defined(cumpsgemm::info_env_name, [&]() {
        int flag;
        cutf::memory::copy(&flag,
                           cuMpSGEMM_get_internal_global_handle()
                                   ->dynamic_launch_handle->flag_buffer +
                               dynamic_launch_id,
                           1);
        const auto gemm_mode =
            cumpsgemm::dynamic_launch::utils::get_gemm_flag(flag);
        const auto scale_A =
            cumpsgemm::dynamic_launch::utils::get_scale_A_flag(flag);
        const auto scale_B =
            cumpsgemm::dynamic_launch::utils::get_scale_B_flag(flag);
        const auto loss_rate_A = cumpsgemm::get_exp_stats(
            cuMpSGEMM_get_internal_global_handle(), A_exp_stats_id);
        const auto loss_rate_B = cumpsgemm::get_exp_stats(
            cuMpSGEMM_get_internal_global_handle(), B_exp_stats_id);
        cuMpSGEMM_log(
            std::string("AUTO[ignore<") +
            get_XeY_format_string(cuMpSGEMM_get_internal_global_handle()
                                      ->exp_stats_handle->ignore_threshold) +
            ", uf<" +
            get_XeY_format_string(cuMpSGEMM_get_internal_global_handle()
                                      ->exp_stats_handle->underflow_threshold) +
            ", tolerance=" +
            get_XeY_format_string(
                cuMpSGEMM_get_internal_global_handle()
                    ->exp_stats_handle->underflow_tolerance_rate) +
            "]: GEMM_MODE=" +
            cuMpSGEMM_get_compute_mode_string(
                (cuMpSGEMM_compute_mode_t)gemm_mode) +
            ", loss_A=" + std::to_string(loss_rate_A.first) + "/" +
            std::to_string(loss_rate_A.second) + "(" +
            std::to_string(static_cast<double>(loss_rate_A.first) /
                           loss_rate_A.second) +
            "), scale_A=" + std::to_string(scale_A) +
            ", loss_B=" + std::to_string(loss_rate_B.first) + "/" +
            std::to_string(loss_rate_B.second) + "(" +
            std::to_string(static_cast<double>(loss_rate_B.first) /
                           loss_rate_B.second) +
            "), scale_B=" + std::to_string(scale_B));
      });

      // Scaling
      cumpsgemm::dynamic_scaling::scale_A(
          cuMpSGEMM_get_internal_global_handle(), (op_A == CUBLAS_OP_N ? m : k),
          (op_A == CUBLAS_OP_N ? k : m), const_cast<T *>(a_dmem_ptr), lda, 0, 1,
          A_exp_stats_id, dynamic_launch_id);
      cumpsgemm::dynamic_scaling::scale_B(
          cuMpSGEMM_get_internal_global_handle(), (op_B == CUBLAS_OP_N ? k : n),
          (op_B == CUBLAS_OP_N ? n : k), const_cast<T *>(b_dmem_ptr), ldb, 0, 1,
          B_exp_stats_id, dynamic_launch_id);

      // Enable dynamic launch
      cumpsgemm::dynamic_launch::set_dynamic_launch_flag_buffer_id(
          cuMpSGEMM_get_internal_global_handle(), dynamic_launch_id);
    } else if (compute_mode == CUMPSGEMM_FP16TCEC_SCALING) {
      // Force execution mode
      dynamic_launch_id = 1;

      cumpsgemm::exp_stats::exp_max_ext(
          cuMpSGEMM_get_internal_global_handle(), (op_A == CUBLAS_OP_N ? m : k),
          (op_A == CUBLAS_OP_N ? k : m), a_dmem_ptr, lda, 1, 0);
      A_exp_stats_id = cumpsgemm::exp_stats::get_current_exp_stats_buffer_id(
          cuMpSGEMM_get_internal_global_handle());
      cumpsgemm::exp_stats::exp_max_ext(
          cuMpSGEMM_get_internal_global_handle(), (op_B == CUBLAS_OP_N ? k : n),
          (op_B == CUBLAS_OP_N ? n : k), b_dmem_ptr, ldb, 1, 0);
      B_exp_stats_id = cumpsgemm::exp_stats::get_current_exp_stats_buffer_id(
          cuMpSGEMM_get_internal_global_handle());

      // Scaling
      cumpsgemm::dynamic_scaling::scale_A(
          cuMpSGEMM_get_internal_global_handle(), (op_A == CUBLAS_OP_N ? m : k),
          (op_A == CUBLAS_OP_N ? k : m), const_cast<T *>(a_dmem_ptr), lda, 0, 1,
          A_exp_stats_id, dynamic_launch_id);
      cumpsgemm::dynamic_scaling::scale_B(
          cuMpSGEMM_get_internal_global_handle(), (op_B == CUBLAS_OP_N ? k : n),
          (op_B == CUBLAS_OP_N ? n : k), const_cast<T *>(b_dmem_ptr), ldb, 0, 1,
          B_exp_stats_id, dynamic_launch_id);
    }

    res = cumpsgemm::gemm<T>(
        cuMpSGEMM_get_internal_global_handle(), op_A, op_B, m, n, k, alpha,
        a_dmem_ptr, lda, b_dmem_ptr, ldb, beta, c_dmem_ptr, ldc,
        compute_mode == CUMPSGEMM_FP16TCEC_SCALING ? CUMPSGEMM_FP16TCEC
                                                   : compute_mode);

    if (compute_mode == CUMPSGEMM_AUTO ||
        compute_mode == CUMPSGEMM_FP16TCEC_SCALING) {
      cumpsgemm::dynamic_scaling::scale_C(
          cuMpSGEMM_get_internal_global_handle(), m, n, c_dmem_ptr, ldc, 0, 1,
          A_exp_stats_id, B_exp_stats_id, dynamic_launch_id);

      // restore A and B
      if (restore_AB) {
        cumpsgemm::dynamic_scaling::reset_scale_A(
            cuMpSGEMM_get_internal_global_handle(),
            (op_A == CUBLAS_OP_N ? m : k), (op_A == CUBLAS_OP_N ? k : m),
            const_cast<T *>(a_dmem_ptr), lda, 0, 1, A_exp_stats_id,
            dynamic_launch_id);
        cumpsgemm::dynamic_scaling::reset_scale_B(
            cuMpSGEMM_get_internal_global_handle(),
            (op_B == CUBLAS_OP_N ? k : n), (op_B == CUBLAS_OP_N ? n : k),
            const_cast<T *>(b_dmem_ptr), ldb, 0, 1, B_exp_stats_id,
            dynamic_launch_id);
      }

      cumpsgemm::dynamic_launch::unset_dynamic_launch_flag_buffer_id(
          cuMpSGEMM_get_internal_global_handle());
    }

    if (profiling_flag) {
      // Record end rimestamp
      cumpsgemm::CULiP::launch_function(cuda_stream,
                                        &cumpsgemm::CULiP::record_timestamp,
                                        (void *)&profile_result.end_timestamp);

      // Print result
      cumpsgemm::CULiP::launch_function(cuda_stream,
                                        &cumpsgemm::CULiP::print_profile_result,
                                        (void *)&profile_result);
    }
  }

  return res;
}

template <class T>
cublasStatus_t cuMpSGEMM_stridedBatched_hijack_core(
    const char *const func_name, cublasHandle_t const cublas_handle,
    const cublasOperation_t op_A, const cublasOperation_t op_B,
    const uint64_t m, const uint64_t n, const uint64_t k, const T *alpha,
    const T *const a_dmem_ptr, const uint64_t lda, const uint64_t stridea,
    const T *const b_dmem_ptr, const uint64_t ldb, const uint64_t strideb,
    const T *beta, T *const c_dmem_ptr, const uint64_t ldc,
    const uint64_t stridec, const uint64_t batch_count) {
  cudaStream_t cuda_stream;
  cublasGetStream(cublas_handle, &cuda_stream);

  if (m == 0 || n == 0 || k == 0 || lda == 0 || ldb == 0 || ldc == 0 ||
      batch_count == 0) {
    return CUBLAS_STATUS_INVALID_VALUE;
  }

  cumpsgemm::CULiP::profile_result profile_result;
  const auto profiling_flag = cumpsgemm::CULiP::is_profiling_enabled();

  cuMpSGEMM_compute_mode_t compute_mode = cuMpSGEMM_get_compute_mode_internal(
      func_name, cublas_handle, op_A, op_B, m, n, k);

  cuMpSGEMM_log(std::string(func_name) + " op=(" + get_cublas_op_str(op_A) +
                ", " + get_cublas_op_str(op_B) + "), shape=(" +
                std::to_string(m) + ", " + std::to_string(n) + ", " +
                std::to_string(k) + "), batch=" + std::to_string(batch_count) +
                ", mode=" + cuMpSGEMM_get_compute_mode_string(compute_mode) +
                "[" + (hijack_mode == dynamic_mode ? "dynamic" : "static") +
                "][exp_stats:" +
                (cumpsgemm::hijack_control::get_internal_global_handle()
                         ->exp_stats_handle->enabled
                     ? "1"
                     : "0") +
                "]");

  cumpsgemm::hijack_control::set_last_called_function_str(
      std::string(func_name) + "," + get_cublas_op_str(op_A) + "," +
      get_cublas_op_str(op_B) + "," + std::to_string(m) + "," +
      std::to_string(n) + "," + std::to_string(k) + "," +
      std::to_string(batch_count) + "," +
      cuMpSGEMM_get_compute_mode_string(compute_mode));

  if (compute_mode == CUMPSGEMM_DRY_RUN) {
    return CUBLAS_STATUS_SUCCESS;
  }

  cublasStatus_t res;

  // -----------------------------------
  // gemm_Mx2x2
  // -----------------------------------
  if (((m & (m - 1)) == 0) && n == 2 && k == 2 && is_gemm_Mx2x2_enabled()) {

    if (profiling_flag) {
      const std::string func_name =
          std::string(std::is_same<T, float>::value ? "s" : "c") +
          "gemm_strided_batch_Mx2x2";
      snprintf(profile_result.function_name,
               profile_result.function_name_length - 1,
               "%s-%s%s-m%lu-n%lu-k%lu-batchCount%lu", func_name.c_str(),
               cumpsgemm::CULiP::get_cublasOperation_t_string(op_A),
               cumpsgemm::CULiP::get_cublasOperation_t_string(op_B), m, n, k,
               batch_count);
      cumpsgemm::CULiP::launch_function(
          cuda_stream, &cumpsgemm::CULiP::record_timestamp,
          (void *)&profile_result.start_timestamp);
    }
    cuMpSGEMM_log(" +---> gemm_Mx2x2");

    mtk::cugemm::gemm_strided_batch_Mx2x2(
        op_A, op_B, m, *alpha, a_dmem_ptr, lda, stridea, b_dmem_ptr, ldb,
        strideb, *beta, c_dmem_ptr, ldc, stridec, batch_count, cuda_stream);

    if (profiling_flag) {
      // Record end rimestamp
      cumpsgemm::CULiP::launch_function(cuda_stream,
                                        &cumpsgemm::CULiP::record_timestamp,
                                        (void *)&profile_result.end_timestamp);

      // Print result
      cumpsgemm::CULiP::launch_function(cuda_stream,
                                        &cumpsgemm::CULiP::print_profile_result,
                                        (void *)&profile_result);
    }

    return CUBLAS_STATUS_SUCCESS;
  }

  // -----------------------------------
  // gemm_2xNx2
  // -----------------------------------
  if (((n & (n - 1)) == 0) && m == 2 && k == 2 && is_gemm_Mx2x2_enabled()) {

    if (profiling_flag) {
      const std::string func_name =
          std::string(std::is_same<T, float>::value ? "s" : "c") +
          "gemm_strided_batch_2xNx2";
      snprintf(profile_result.function_name,
               profile_result.function_name_length - 1,
               "%s-%s%s-m%lu-n%lu-k%lu-batchCount%lu", func_name.c_str(),
               cumpsgemm::CULiP::get_cublasOperation_t_string(op_A),
               cumpsgemm::CULiP::get_cublasOperation_t_string(op_B), m, n, k,
               batch_count);
      cumpsgemm::CULiP::launch_function(
          cuda_stream, &cumpsgemm::CULiP::record_timestamp,
          (void *)&profile_result.start_timestamp);
    }
    cuMpSGEMM_log(" +---> gemm_2xNx2");

    mtk::cugemm::gemm_strided_batch_2xNx2(
        op_A, op_B, n, *alpha, a_dmem_ptr, lda, stridea, b_dmem_ptr, ldb,
        strideb, *beta, c_dmem_ptr, ldc, stridec, batch_count, cuda_stream);

    if (profiling_flag) {
      // Record end rimestamp
      cumpsgemm::CULiP::launch_function(cuda_stream,
                                        &cumpsgemm::CULiP::record_timestamp,
                                        (void *)&profile_result.end_timestamp);

      // Print result
      cumpsgemm::CULiP::launch_function(cuda_stream,
                                        &cumpsgemm::CULiP::print_profile_result,
                                        (void *)&profile_result);
    }

    return CUBLAS_STATUS_SUCCESS;
  }

  if (compute_mode == CUMPSGEMM_CUBLAS ||
      compute_mode == CUMPSGEMM_CUBLAS_FP16TC ||
      compute_mode == CUMPSGEMM_CUBLAS_TF32TC ||
      compute_mode == CUMPSGEMM_CUBLAS_SIMT) {
    // -----------------------------------
    // cuBLAS
    // -----------------------------------
    cublasGemmAlgo_t gemm_algo = CUBLAS_GEMM_DEFAULT;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    const cudaDataType_t io_datat_type = get_cuda_data_type<T>();
    if (compute_mode == CUMPSGEMM_CUBLAS_TF32TC) {
      gemm_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
      compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    } else if (compute_mode == CUMPSGEMM_CUBLAS_FP16TC) {
      gemm_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
      compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
    } else if (compute_mode == CUMPSGEMM_CUBLAS_SIMT) {
      // Do nothing
    } else {
      const std::string func_name_str(func_name);
      if (func_name_str == "cublasSgemmStridedBatched" ||
          func_name_str == "cublasCgemmStridedBatched") {
        cublasMath_t math_mode;
        cublasGetMathMode(cublas_handle, &math_mode);
        switch (math_mode) {
        case CUBLAS_DEFAULT_MATH:
        case CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION:
          // Do nothing
          break;
        case CUBLAS_TF32_TENSOR_OP_MATH:
          gemm_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
          compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
          break;
        case CUBLAS_TENSOR_OP_MATH:
          gemm_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
          compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
          break;
        default:
          break;
        }
      }
    }

    cublasStatus_t (*func_ptr)(
        cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
        const void *, const void *, cudaDataType, int, long long int,
        const void *, cudaDataType, int, long long int, const void *, void *,
        cudaDataType, int, long long int, int, cublasComputeType_t,
        cublasGemmAlgo_t);

    *(void **)(&func_ptr) =
        cuMpSGEMM_get_function_pointer("cublasGemmStridedBatchedEx");
    if (func_ptr == nullptr) {
      cuMpSGEMM_error(std::string("Could not load the cuBLAS function \"") +
                      func_name + "\"");
    }

    if (profiling_flag) {
      snprintf(profile_result.function_name,
               profile_result.function_name_length - 1,
               "%s-%s%s-m%lu-n%lu-k%lu-batchCount%lu", func_name,
               cumpsgemm::CULiP::get_cublasOperation_t_string(op_A),
               cumpsgemm::CULiP::get_cublasOperation_t_string(op_B), m, n, k,
               batch_count);
      cumpsgemm::CULiP::launch_function(
          cuda_stream, &cumpsgemm::CULiP::record_timestamp,
          (void *)&profile_result.start_timestamp);
    }

    res = (*func_ptr)(cublas_handle, op_A, op_B, m, n, k, alpha, a_dmem_ptr,
                      io_datat_type, lda, stridea, b_dmem_ptr, io_datat_type,
                      ldb, strideb, beta, c_dmem_ptr, io_datat_type, ldc,
                      stridec, batch_count, compute_type, gemm_algo);

    if (profiling_flag) {
      // Record end rimestamp
      cumpsgemm::CULiP::launch_function(cuda_stream,
                                        &cumpsgemm::CULiP::record_timestamp,
                                        (void *)&profile_result.end_timestamp);

      // Print result
      cumpsgemm::CULiP::launch_function(cuda_stream,
                                        &cumpsgemm::CULiP::print_profile_result,
                                        (void *)&profile_result);
    }
  } else {
    // -----------------------------------
    // cuMpSGEMM
    // -----------------------------------
    if (profiling_flag) {
      const std::string func_name =
          std::string(std::is_same<T, float>::value ? "s" : "c") +
          "gemm_stridedBatch_" +
          std::string(cuMpSGEMM_get_compute_mode_string(compute_mode));
      snprintf(profile_result.function_name,
               profile_result.function_name_length - 1,
               "%s-%s%s-m%lu-n%lu-k%lu-batchCount%lu", func_name.c_str(),
               cumpsgemm::CULiP::get_cublasOperation_t_string(op_A),
               cumpsgemm::CULiP::get_cublasOperation_t_string(op_B), m, n, k,
               batch_count);
      cumpsgemm::CULiP::launch_function(
          cuda_stream, &cumpsgemm::CULiP::record_timestamp,
          (void *)&profile_result.start_timestamp);
    }

    unsigned A_exp_stats_id, B_exp_stats_id, dynamic_launch_id;
    if (compute_mode == CUMPSGEMM_AUTO) {
      // Exp stats
      cumpsgemm::exp_stats::exp_stats_ext(
          cuMpSGEMM_get_internal_global_handle(), (op_A == CUBLAS_OP_N ? m : k),
          (op_A == CUBLAS_OP_N ? k : m), a_dmem_ptr, lda, batch_count, stridea);
      A_exp_stats_id = cumpsgemm::exp_stats::get_current_exp_stats_buffer_id(
          cuMpSGEMM_get_internal_global_handle());
      cumpsgemm::exp_stats::exp_stats_ext(
          cuMpSGEMM_get_internal_global_handle(), (op_B == CUBLAS_OP_N ? k : n),
          (op_B == CUBLAS_OP_N ? n : k), b_dmem_ptr, ldb, batch_count, strideb);
      B_exp_stats_id = cumpsgemm::exp_stats::get_current_exp_stats_buffer_id(
          cuMpSGEMM_get_internal_global_handle());

      // Kernel decision
      dynamic_launch_id =
          cumpsgemm::dynamic_launch::get_next_dynamic_launch_flag_buffer_id(
              cuMpSGEMM_get_internal_global_handle());
      cumpsgemm::dynamic_scaling::set_dynamic_launch_buffer_by_exp_stats(
          cuMpSGEMM_get_internal_global_handle(), dynamic_launch_id,
          A_exp_stats_id, B_exp_stats_id);

      cuMpSGEMM_run_if_env_defined(cumpsgemm::info_env_name, [&]() {
        int flag;
        cutf::memory::copy(&flag,
                           cuMpSGEMM_get_internal_global_handle()
                                   ->dynamic_launch_handle->flag_buffer +
                               dynamic_launch_id,
                           1);
        const auto gemm_mode =
            cumpsgemm::dynamic_launch::utils::get_gemm_flag(flag);
        const auto scale_A =
            cumpsgemm::dynamic_launch::utils::get_scale_A_flag(flag);
        const auto scale_B =
            cumpsgemm::dynamic_launch::utils::get_scale_B_flag(flag);
        const auto loss_rate_A = cumpsgemm::get_exp_stats(
            cuMpSGEMM_get_internal_global_handle(), A_exp_stats_id);
        const auto loss_rate_B = cumpsgemm::get_exp_stats(
            cuMpSGEMM_get_internal_global_handle(), B_exp_stats_id);
        cuMpSGEMM_log(
            std::string("AUTO[ignore<") +
            get_XeY_format_string(cuMpSGEMM_get_internal_global_handle()
                                      ->exp_stats_handle->ignore_threshold) +
            ", uf<" +
            get_XeY_format_string(cuMpSGEMM_get_internal_global_handle()
                                      ->exp_stats_handle->underflow_threshold) +
            ", tolerance=" +
            get_XeY_format_string(
                cuMpSGEMM_get_internal_global_handle()
                    ->exp_stats_handle->underflow_tolerance_rate) +
            "]: GEMM_MODE=" +
            cuMpSGEMM_get_compute_mode_string(
                (cuMpSGEMM_compute_mode_t)gemm_mode) +
            ", loss_A=" + std::to_string(loss_rate_A.first) + "/" +
            std::to_string(loss_rate_A.second) + "(" +
            std::to_string(static_cast<double>(loss_rate_A.first) /
                           loss_rate_A.second) +
            "), scale_A=" + std::to_string(scale_A) +
            ", loss_B=" + std::to_string(loss_rate_B.first) + "/" +
            std::to_string(loss_rate_B.second) + "(" +
            std::to_string(static_cast<double>(loss_rate_B.first) /
                           loss_rate_B.second) +
            "), scale_B=" + std::to_string(scale_B));
      });

      // Scaling
      cumpsgemm::dynamic_scaling::scale_A(
          cuMpSGEMM_get_internal_global_handle(), (op_A == CUBLAS_OP_N ? m : k),
          (op_A == CUBLAS_OP_N ? k : m), const_cast<T *>(a_dmem_ptr), lda,
          stridea, batch_count, A_exp_stats_id, dynamic_launch_id);
      cumpsgemm::dynamic_scaling::scale_B(
          cuMpSGEMM_get_internal_global_handle(), (op_B == CUBLAS_OP_N ? k : n),
          (op_B == CUBLAS_OP_N ? n : k), const_cast<T *>(b_dmem_ptr), ldb,
          strideb, batch_count, B_exp_stats_id, dynamic_launch_id);

      // Enable dynamic launch
      cumpsgemm::dynamic_launch::set_dynamic_launch_flag_buffer_id(
          cuMpSGEMM_get_internal_global_handle(), dynamic_launch_id);
    } else if (compute_mode == CUMPSGEMM_FP16TCEC_SCALING) {
      // Force execution mode
      dynamic_launch_id = 1;

      // Exp stats
      cumpsgemm::exp_stats::exp_max_ext(
          cuMpSGEMM_get_internal_global_handle(), (op_A == CUBLAS_OP_N ? m : k),
          (op_A == CUBLAS_OP_N ? k : m), a_dmem_ptr, lda, batch_count, stridea);
      A_exp_stats_id = cumpsgemm::exp_stats::get_current_exp_stats_buffer_id(
          cuMpSGEMM_get_internal_global_handle());
      cumpsgemm::exp_stats::exp_max_ext(
          cuMpSGEMM_get_internal_global_handle(), (op_B == CUBLAS_OP_N ? k : n),
          (op_B == CUBLAS_OP_N ? n : k), b_dmem_ptr, ldb, batch_count, strideb);
      B_exp_stats_id = cumpsgemm::exp_stats::get_current_exp_stats_buffer_id(
          cuMpSGEMM_get_internal_global_handle());

      // Scaling
      cumpsgemm::dynamic_scaling::scale_A(
          cuMpSGEMM_get_internal_global_handle(), (op_A == CUBLAS_OP_N ? m : k),
          (op_A == CUBLAS_OP_N ? k : m), const_cast<T *>(a_dmem_ptr), lda,
          stridea, batch_count, A_exp_stats_id, dynamic_launch_id);
      cumpsgemm::dynamic_scaling::scale_B(
          cuMpSGEMM_get_internal_global_handle(), (op_B == CUBLAS_OP_N ? k : n),
          (op_B == CUBLAS_OP_N ? n : k), const_cast<T *>(b_dmem_ptr), ldb,
          strideb, batch_count, B_exp_stats_id, dynamic_launch_id);
    }

    res = cumpsgemm::gemm_stridedBatch<T>(
        cuMpSGEMM_get_internal_global_handle(), op_A, op_B, m, n, k, alpha,
        a_dmem_ptr, lda, stridea, b_dmem_ptr, ldb, strideb, beta, c_dmem_ptr,
        ldc, stridec, batch_count,
        compute_mode == CUMPSGEMM_FP16TCEC_SCALING ? CUMPSGEMM_FP16TCEC
                                                   : compute_mode);

    if (compute_mode == CUMPSGEMM_AUTO ||
        compute_mode == CUMPSGEMM_FP16TCEC_SCALING) {
      cumpsgemm::dynamic_scaling::scale_C(
          cuMpSGEMM_get_internal_global_handle(), m, n, c_dmem_ptr, ldc,
          stridec, batch_count, A_exp_stats_id, B_exp_stats_id,
          dynamic_launch_id);

      // restore A and B
      if (restore_AB) {
        cumpsgemm::dynamic_scaling::reset_scale_A(
            cuMpSGEMM_get_internal_global_handle(),
            (op_A == CUBLAS_OP_N ? m : k), (op_A == CUBLAS_OP_N ? k : m),
            const_cast<T *>(a_dmem_ptr), lda, stridea, batch_count,
            A_exp_stats_id, dynamic_launch_id);
        cumpsgemm::dynamic_scaling::reset_scale_B(
            cuMpSGEMM_get_internal_global_handle(),
            (op_B == CUBLAS_OP_N ? k : n), (op_B == CUBLAS_OP_N ? n : k),
            const_cast<T *>(b_dmem_ptr), ldb, strideb, batch_count,
            B_exp_stats_id, dynamic_launch_id);
      }
    }

    if (profiling_flag) {
      // Record end rimestamp
      cumpsgemm::CULiP::launch_function(cuda_stream,
                                        &cumpsgemm::CULiP::record_timestamp,
                                        (void *)&profile_result.end_timestamp);

      // Print result
      cumpsgemm::CULiP::launch_function(cuda_stream,
                                        &cumpsgemm::CULiP::print_profile_result,
                                        (void *)&profile_result);
    }
  }
  return res;
}

// cuBLAS functions
extern "C" {
CUBLASAPI cublasStatus_t
cublasSgemm_v2(cublasHandle_t cublas_handle, cublasOperation_t op_A,
               cublasOperation_t op_B, int m, int n, int k, const float *alpha,
               const float *a_dmem_ptr, int lda, const float *b_dmem_ptr,
               int ldb, const float *beta, float *c_dmem_ptr, int ldc) {
#ifdef __CUDA_ARCH__
  return CUBLAS_STATUS_NOT_SUPPORTED;
#else
  return cuMpSGEMM_hijack_core<float>(__func__, cublas_handle, op_A, op_B, m, n,
                                      k, alpha, a_dmem_ptr, lda, b_dmem_ptr,
                                      ldb, beta, c_dmem_ptr, ldc);
#endif
}

CUBLASAPI cublasStatus_t cublasCgemm_v2(
    cublasHandle_t cublas_handle, cublasOperation_t op_A,
    cublasOperation_t op_B, int m, int n, int k, const cuComplex *alpha,
    const cuComplex *a_dmem_ptr, int lda, const cuComplex *b_dmem_ptr, int ldb,
    const cuComplex *beta, cuComplex *c_dmem_ptr, int ldc) {
#ifdef __CUDA_ARCH__
  return CUBLAS_STATUS_NOT_SUPPORTED;
#else
  return cuMpSGEMM_hijack_core<cuComplex>(
      __func__, cublas_handle, op_A, op_B, m, n, k, alpha, a_dmem_ptr, lda,
      b_dmem_ptr, ldb, beta, c_dmem_ptr, ldc);
#endif
}

CUBLASAPI cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t cublas_handle, cublasOperation_t op_A,
    cublasOperation_t op_B, int m, int n, int k, const float *alpha,
    const float *a_dmem_ptr, int lda, long long int stridea,
    const float *b_dmem_ptr, int ldb, long long int strideb, const float *beta,
    float *c_dmem_ptr, int ldc, long long int stridec, const int batch_count) {
#ifdef __CUDA_ARCH__
  return CUBLAS_STATUS_NOT_SUPPORTED;
#else
  return cuMpSGEMM_stridedBatched_hijack_core<float>(
      __func__, cublas_handle, op_A, op_B, m, n, k, alpha, a_dmem_ptr, lda,
      stridea, b_dmem_ptr, ldb, strideb, beta, c_dmem_ptr, ldc, stridec,
      batch_count);
#endif
}

CUBLASAPI cublasStatus_t cublasCgemmStridedBatched(
    cublasHandle_t cublas_handle, cublasOperation_t op_A,
    cublasOperation_t op_B, int m, int n, int k, const cuComplex *alpha,
    const cuComplex *a_dmem_ptr, int lda, const long long int stridea,
    const cuComplex *b_dmem_ptr, int ldb, const long long int strideb,
    const cuComplex *beta, cuComplex *c_dmem_ptr, int ldc,
    const long long int stridec, const int batch_count) {
#ifdef __CUDA_ARCH__
  return CUBLAS_STATUS_NOT_SUPPORTED;
#else
  return cuMpSGEMM_stridedBatched_hijack_core<cuComplex>(
      __func__, cublas_handle, op_A, op_B, m, n, k, alpha, a_dmem_ptr, lda,
      stridea, b_dmem_ptr, ldb, strideb, beta, c_dmem_ptr, ldc, stridec,
      batch_count);
#endif
}

CUBLASAPI cublasStatus_t cublasGemmEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void *alpha, const void *A, cudaDataType_t Atype,
    int lda, const void *B, cudaDataType_t Btype, int ldb, const void *beta,
    void *C, cudaDataType_t Ctype, int ldc, cublasComputeType_t computeType,
    cublasGemmAlgo_t algo) {
#ifdef __CUDA_ARCH__
  return CUBLAS_STATUS_NOT_SUPPORTED;
#else
  if (Atype == CUDA_R_32F && Btype == CUDA_R_32F && Ctype == CUDA_R_32F) {
    return cuMpSGEMM_hijack_core<float>(__func__, handle, transa, transb, m, n,
                                        k,
                                        reinterpret_cast<const float *>(alpha),
                                        reinterpret_cast<const float *>(A), lda,
                                        reinterpret_cast<const float *>(B), ldb,
                                        reinterpret_cast<const float *>(beta),
                                        reinterpret_cast<float *>(C), ldc);
  }
  if (Atype == CUDA_C_32F && Btype == CUDA_C_32F && Ctype == CUDA_C_32F) {
    return cuMpSGEMM_hijack_core<cuComplex>(
        __func__, handle, transa, transb, m, n, k,
        reinterpret_cast<const cuComplex *>(alpha),
        reinterpret_cast<const cuComplex *>(A), lda,
        reinterpret_cast<const cuComplex *>(B), ldb,
        reinterpret_cast<const cuComplex *>(beta),
        reinterpret_cast<cuComplex *>(C), ldc);
  }

  cudaStream_t cuda_stream;
  cublasGetStream(handle, &cuda_stream);

  cumpsgemm::CULiP::profile_result profile_result;
  const auto profiling_flag = cumpsgemm::CULiP::is_profiling_enabled();

  cublasStatus_t (*func_ptr)(
      cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
      const void *, const void *, cudaDataType_t, int, const void *,
      cudaDataType_t, int, const void *, void *, cudaDataType_t, int,
      cublasComputeType_t, cublasGemmAlgo_t);
  *(void **)(&func_ptr) = cuMpSGEMM_get_function_pointer(__func__);

  if (profiling_flag) {
    snprintf(profile_result.function_name,
             profile_result.function_name_length - 1, "%s-%s%s-m%d-n%d-k%d",
             __func__, cumpsgemm::CULiP::get_cublasOperation_t_string(transa),
             cumpsgemm::CULiP::get_cublasOperation_t_string(transb), m, n, k);
    cumpsgemm::CULiP::launch_function(cuda_stream,
                                      &cumpsgemm::CULiP::record_timestamp,
                                      (void *)&profile_result.start_timestamp);
  }

  const auto res =
      (*func_ptr)(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B,
                  Btype, ldb, beta, C, Ctype, ldc, computeType, algo);

  if (profiling_flag) {
    // Record end rimestamp
    cumpsgemm::CULiP::launch_function(cuda_stream,
                                      &cumpsgemm::CULiP::record_timestamp,
                                      (void *)&profile_result.end_timestamp);

    // Print result
    cumpsgemm::CULiP::launch_function(cuda_stream,
                                      &cumpsgemm::CULiP::print_profile_result,
                                      (void *)&profile_result);
  }

  return res;
#endif
}

CUBLASAPI cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void *alpha, const void *A, cudaDataType_t Atype,
    int lda, long long int strideA, const void *B, cudaDataType_t Btype,
    int ldb, long long int strideB, const void *beta, void *C,
    cudaDataType_t Ctype, int ldc, long long int strideC, int batch_count,
    cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
#ifdef __CUDA_ARCH__
  return CUBLAS_STATUS_NOT_SUPPORTED;
#else
  if (Atype == CUDA_R_32F && Btype == CUDA_R_32F && Ctype == CUDA_R_32F) {
    return cuMpSGEMM_stridedBatched_hijack_core<float>(
        __func__, handle, transa, transb, m, n, k,
        reinterpret_cast<const float *>(alpha),
        reinterpret_cast<const float *>(A), lda, strideA,
        reinterpret_cast<const float *>(B), ldb, strideB,
        reinterpret_cast<const float *>(beta), reinterpret_cast<float *>(C),
        ldc, strideC, batch_count);
  }
  if (Atype == CUDA_C_32F && Btype == CUDA_C_32F && Ctype == CUDA_C_32F) {
    return cuMpSGEMM_stridedBatched_hijack_core<cuComplex>(
        __func__, handle, transa, transb, m, n, k,
        reinterpret_cast<const cuComplex *>(alpha),
        reinterpret_cast<const cuComplex *>(A), lda, strideA,
        reinterpret_cast<const cuComplex *>(B), ldb, strideB,
        reinterpret_cast<const cuComplex *>(beta),
        reinterpret_cast<cuComplex *>(C), ldc, strideC, batch_count);
  }

  cudaStream_t cuda_stream;
  cublasGetStream(handle, &cuda_stream);

  cumpsgemm::CULiP::profile_result profile_result;
  const auto profiling_flag = cumpsgemm::CULiP::is_profiling_enabled();

  cublasStatus_t (*func_ptr)(
      cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
      const void *, const void *, cudaDataType_t, int, long long int,
      const void *, cudaDataType_t, int, long long int, const void *, void *,
      cudaDataType_t, int, long long int, int, cublasComputeType_t,
      cublasGemmAlgo_t);
  *(void **)(&func_ptr) = cuMpSGEMM_get_function_pointer(__func__);

  if (profiling_flag) {
    snprintf(profile_result.function_name,
             profile_result.function_name_length - 1,
             "%s-%s%s-m%d-n%d-k%d-batch_count%d", __func__,
             cumpsgemm::CULiP::get_cublasOperation_t_string(transa),
             cumpsgemm::CULiP::get_cublasOperation_t_string(transb), m, n, k,
             batch_count);
    cumpsgemm::CULiP::launch_function(cuda_stream,
                                      &cumpsgemm::CULiP::record_timestamp,
                                      (void *)&profile_result.start_timestamp);
  }

  const auto res =
      (*func_ptr)(handle, transa, transb, m, n, k, alpha, A, Atype, lda,
                  strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC,
                  batch_count, computeType, algo);

  if (profiling_flag) {
    // Record end rimestamp
    cumpsgemm::CULiP::launch_function(cuda_stream,
                                      &cumpsgemm::CULiP::record_timestamp,
                                      (void *)&profile_result.end_timestamp);

    // Print result
    cumpsgemm::CULiP::launch_function(cuda_stream,
                                      &cumpsgemm::CULiP::print_profile_result,
                                      (void *)&profile_result);
  }

  return res;
#endif
}
} // extern "C"

cuMpSGEMM_handle *cumpsgemm::hijack_control::get_internal_global_handle() {
  return cuMpSGEMM_get_internal_global_handle();
}

void cumpsgemm::hijack_control::set_compute_mode(
    const cuMpSGEMM_compute_mode_t mode) {
  if (mode == CUMPSGEMM_FP32_SIMT) {
    throw std::runtime_error(
        "CUMPSGEMM_FP32_SIMT mode is currently not supported.");
  }
  internal_global_compute_mode = mode;
  hijack_mode = static_mode;
}

void cumpsgemm::hijack_control::unset_compute_mode() {
  hijack_mode = dynamic_mode;
}

void cumpsgemm::hijack_control::set_exp_stats_params(
    const float ignore_threshold, const float underflow_threshold,
    const float underflow_tolerance_rate) {
  cuMpSGEMM_log("AUTO config: ignore_threshold=" +
                get_XeY_format_string(ignore_threshold) + " @" +
                std::string(__func__));
  cuMpSGEMM_log("AUTO config: underflow_threshold=" +
                get_XeY_format_string(underflow_threshold) + " @" +
                std::string(__func__));
  cuMpSGEMM_log("AUTO config: underflow_tolerance_rate=" +
                get_XeY_format_string(underflow_tolerance_rate) + " @" +
                std::string(__func__));

  cumpsgemm::set_exp_stats_params(get_internal_global_handle(),
                                  ignore_threshold, underflow_threshold,
                                  underflow_tolerance_rate);
}

void cumpsgemm::hijack_control::reset_exp_stats_buffer_id() {
  cumpsgemm::exp_stats::reset_exp_stats_buffer_id(get_internal_global_handle());
}

std::string cumpsgemm::hijack_control::get_last_called_function_str() {
  return internal_global_last_called_function_str;
}

void cumpsgemm::hijack_control::set_last_called_function_str(
    const std::string func_str) {
  internal_global_last_called_function_str = func_str;
}

void cumpsgemm::hijack_control::clear_last_called_function_str() {
  cumpsgemm::hijack_control::set_last_called_function_str("");
}

void cumpsgemm::hijack_control::enable_custom_gemm_Mx2x2() {
  global_internal_gemm_Mx2x2_enabled = true;
}

void cumpsgemm::hijack_control::disable_custom_gemm_Mx2x2() {
  global_internal_gemm_Mx2x2_enabled = false;
}

void cumpsgemm::hijack_control::enable_restoring_AB_after_scaling() {
  restore_AB = true;
  cuMpSGEMM_log("AUTO config: restore_AB_scaling=True @" +
                std::string(__func__));
}

void cumpsgemm::hijack_control::disable_restoring_AB_after_scaling() {
  restore_AB = false;
  cuMpSGEMM_log("AUTO config: restore_AB_scaling=False @" +
                std::string(__func__));
}

bool cumpsgemm::hijack_control::is_library_loaded() { return true; }

void cumpsgemm::hijack_control::set_control_function(
    const cumpsgemm::hijack_control::control_function_t control_func) {
  internal_global_control_func = control_func;
}

void cumpsgemm::hijack_control::unset_control_function() {
  internal_global_control_func = 0;
}
