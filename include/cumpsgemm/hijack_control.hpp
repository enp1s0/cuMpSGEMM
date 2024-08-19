#pragma once
#include "detail/common.h"
#include <functional>
#include <string>
#include <vector>

namespace cumpsgemm {
namespace hijack_control {
cuMpSGEMM_handle_t get_internal_global_handle();

void set_compute_mode(const cuMpSGEMM_compute_mode_t mode);
void unset_compute_mode();
void enable_custom_gemm_Mx2x2();
void disable_custom_gemm_Mx2x2();

void reset_exp_stats_buffer_id();
void set_exp_stats_params(const float ignore_threshold,
                          const float underflow_threshold,
                          const float underflow_tolerance_rate);
void enable_restoring_AB_after_scaling();
void disable_restoring_AB_after_scaling();

std::string get_last_called_function_str();
void set_last_called_function_str(const std::string func_str);
void clear_last_called_function_str();

bool is_library_loaded();

using control_function_t = std::function<cuMpSGEMM_compute_mode_t(
    const int, const int, const unsigned, const unsigned, const unsigned)>;
void set_control_function(const control_function_t control_function);
void unset_control_function();
} // namespace hijack_control
} // namespace cumpsgemm
