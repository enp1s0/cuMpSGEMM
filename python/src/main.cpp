#include <cumpsgemm/hijack_control.hpp>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <string>
#include <utility>

int global_auto_kernel_selection_enabled = 0;
unsigned global_cublas_dim_mn_threshold = 128;
unsigned global_cublas_dim_k_threshold = 64;

namespace cumpsgemm {
namespace hijack_control {
void set_compute_mode(const cuMpSGEMM_compute_mode_t mode){};
void unset_compute_mode(){};
void enable_custom_gemm_Mx2x2(){};
void disable_custom_gemm_Mx2x2(){};

void reset_exp_stats_buffer_id(){};
void set_exp_stats_params(const float ignore_threshold,
                          const float underflow_threshold,
                          const float underflow_tolerance_rate){};
void enable_restoring_AB_after_scaling(){};
void disable_restoring_AB_after_scaling(){};

std::string get_last_called_function_str() { return ""; };
void set_last_called_function_str(const std::string func_str){};
void clear_last_called_function_str(){};

bool is_library_loaded() { return false; }

void set_control_function(const control_function_t){};
void unset_control_function(){};
} // namespace hijack_control
} // namespace cumpsgemm

void set_compute_mode(const cuMpSGEMM_compute_mode_t compute_mode) {
  cumpsgemm::hijack_control::set_compute_mode(compute_mode);
}

void unset_compute_mode() { cumpsgemm::hijack_control::unset_compute_mode(); }

void enable_custom_gemm_Mx2x2() {
  cumpsgemm::hijack_control::enable_custom_gemm_Mx2x2();
};
void disable_custom_gemm_Mx2x2() {
  cumpsgemm::hijack_control::disable_custom_gemm_Mx2x2();
};

void set_exp_stats_params(const float ignore_threshold,
                          const float underflow_threshold,
                          const float underflow_tolerance_rate) {
  cumpsgemm::hijack_control::set_exp_stats_params(
      ignore_threshold, underflow_threshold, underflow_tolerance_rate);
}

std::string get_last_called_function_str() {
  return cumpsgemm::hijack_control::get_last_called_function_str();
}

void set_last_called_function_str(const std::string func_str) {
  cumpsgemm::hijack_control::set_last_called_function_str(func_str);
}

void clear_last_called_function_str() {
  cumpsgemm::hijack_control::clear_last_called_function_str();
}

bool is_library_loaded() {
  return cumpsgemm::hijack_control::is_library_loaded();
}

void enable_restoring_AB_after_scaling() {
  cumpsgemm::hijack_control::enable_restoring_AB_after_scaling();
};

void disable_restoring_AB_after_scaling() {
  cumpsgemm::hijack_control::disable_restoring_AB_after_scaling();
};

void set_control_function(
    const cumpsgemm::hijack_control::control_function_t control_func) {
  cumpsgemm::hijack_control::set_control_function(control_func);
}

void unset_control_function() {
  cumpsgemm::hijack_control::unset_control_function();
}

void enable_auto_kernel_selection() {
  global_auto_kernel_selection_enabled = true;
}
void disable_auto_kernel_selection() {
  global_auto_kernel_selection_enabled = false;
}
bool is_auto_kernel_selection_enabled() {
  return global_auto_kernel_selection_enabled;
}
void set_global_cublas_dim_mn_threshold(const unsigned dim) {
  global_cublas_dim_mn_threshold = dim;
}
unsigned get_global_cublas_dim_mn_threshold() {
  return global_cublas_dim_mn_threshold;
}
void set_global_cublas_dim_k_threshold(const unsigned dim) {
  global_cublas_dim_k_threshold = dim;
}
unsigned get_global_cublas_dim_k_threshold() {
  return global_cublas_dim_k_threshold;
}

PYBIND11_MODULE(cumpsgemm_hijack_control, m) {
  m.doc() = "cuMpSGEMM hijack control API";

  m.def("unset_compute_mode", &unset_compute_mode, "unset_compute_mode");
  m.def("set_compute_mode", &set_compute_mode, "set_compute_mode",
        pybind11::arg("compute_mode"));
  m.def("enable_custom_gemm_Mx2x2", &enable_custom_gemm_Mx2x2,
        "enable_custom_gemm_Mx2x2");
  m.def("disable_custom_gemm_Mx2x2", &disable_custom_gemm_Mx2x2,
        "disable_custom_gemm_Mx2x2");
  m.def("set_exp_stats_params", &set_exp_stats_params, "set_exp_stats_params",
        pybind11::arg("ignore_threshold"), pybind11::arg("underflow_threshold"),
        pybind11::arg("underflow_tolerance_rate"));

  m.def("get_last_called_function_str", &get_last_called_function_str,
        "get_last_called_function_str");
  m.def("set_last_called_function_str", &set_last_called_function_str,
        "set_last_called_function_str");
  m.def("clear_last_called_function_str", &clear_last_called_function_str,
        "clear_last_called_function_str");

  m.def("enable_restoring_AB_after_scaling", &enable_restoring_AB_after_scaling,
        "enable_restoring_AB_after_scaling");
  m.def("disable_restoring_AB_after_scaling",
        &disable_restoring_AB_after_scaling,
        "disable_restoring_AB_after_scaling");

  m.def("set_control_function", &set_control_function, "set_control_function",
        pybind11::arg("control_func"));
  m.def("unset_control_function", &unset_control_function,
        "unset_control_function");

  pybind11::enum_<cuMpSGEMM_compute_mode_t>(m, "compute_mode")
      .value("CUMPSGEMM_CUBLAS", CUMPSGEMM_CUBLAS)
      .value("CUMPSGEMM_FP16TCEC", CUMPSGEMM_FP16TCEC)
      .value("CUMPSGEMM_TF32TCEC", CUMPSGEMM_TF32TCEC)
      .value("CUMPSGEMM_FP16TC", CUMPSGEMM_FP16TC)
      .value("CUMPSGEMM_TF32TC", CUMPSGEMM_TF32TC)
      .value("CUMPSGEMM_CUBLAS_SIMT", CUMPSGEMM_CUBLAS_SIMT)
      .value("CUMPSGEMM_CUBLAS_FP16TC", CUMPSGEMM_CUBLAS_FP16TC)
      .value("CUMPSGEMM_CUBLAS_TF32TC", CUMPSGEMM_CUBLAS_TF32TC)
      .value("CUMPSGEMM_DRY_RUN", CUMPSGEMM_DRY_RUN)
      .value("CUMPSGEMM_AUTO", CUMPSGEMM_AUTO)
      .export_values();

  m.def("enable_auto_kernel_selection", &enable_auto_kernel_selection,
        "enable_auto_kernel_selection");
  m.def("disable_auto_kernel_selection", &disable_auto_kernel_selection,
        "disable_auto_kernel_selection");
  m.def("is_auto_kernel_selection_enabled", &is_auto_kernel_selection_enabled,
        "is_auto_kernel_selection_enabled");
  m.def("set_global_cublas_dim_mn_threshold",
        &set_global_cublas_dim_mn_threshold,
        "set_global_cublas_dim_mn_threshold", pybind11::arg("dim"));
  m.def("get_global_cublas_dim_mn_threshold",
        &get_global_cublas_dim_mn_threshold,
        "get_global_cublas_dim_mn_threshold");
  m.def("set_global_cublas_dim_k_threshold", &set_global_cublas_dim_k_threshold,
        "set_global_cublas_dim_k_threshold", pybind11::arg("dim"));
  m.def("get_global_cublas_dim_k_threshold", &get_global_cublas_dim_k_threshold,
        "get_global_cublas_dim_k_threshold");
  m.def("is_library_loaded", &is_library_loaded, "is_library_loaded");
}
