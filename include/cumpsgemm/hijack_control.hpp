#pragma once
#include <string>
#include <vector>
#include "detail/common.h"

namespace cumpsgemm {
namespace hijack_control {
cuMpSGEMM_handle_t get_internal_global_handle();

void set_compute_mode(const cuMpSGEMM_compute_mode_t mode);
void unset_compute_mode();
void enable_custom_gemm_Mx2x2();
void disable_custom_gemm_Mx2x2();

void enable_exp_stats();
void disable_exp_stats();
std::pair<std::size_t, std::size_t> get_exp_stats(const unsigned buffer_id);
unsigned get_current_exp_stats_buffer_id();
void reset_exp_stats_buffer_id();
void set_exp_stats_params(
		const float ignore_threshold,
		const float underflow_threshold,
		const float underflow_tolerance_rate
		);
void exp_stats(
		const unsigned m,
		const unsigned n,
		const float* const ptr,
		const unsigned ld,
		const unsigned batch_size = 1,
		const unsigned stride = 0
		);
bool is_exp_stats_enabled();
std::string get_last_called_function_str();
void set_last_called_function_str(const std::string func_str);
void clear_last_called_function_str();

unsigned get_next_dynamic_launch_flag_buffer_id();
void set_dynamic_launch_flag_buffer_id_use(unsigned id);
void set_dynamic_launch_buffer_by_exp_stats(
		const unsigned dynamic_launch_flag_buffer_id,
		const unsigned exp_stats_buffer_A_id,
		const unsigned exp_stats_buffer_B_id
		);

void scale_A(
		const unsigned exp_stats_buffer_id,
		const unsigned dynamic_launch_flag_buffer_id,
		const unsigned m,
		const unsigned n,
		float* const ptr,
		const unsigned ld,
		const unsigned batch_size = 1,
		const unsigned stride = 0
		);
void scale_B(
		const unsigned exp_stats_buffer_id,
		const unsigned dynamic_launch_flag_buffer_id,
		const unsigned m,
		const unsigned n,
		float* const ptr,
		const unsigned ld,
		const unsigned batch_size = 1,
		const unsigned stride = 0
		);
void scale_C(
		const unsigned exp_stats_buffer_A_id,
		const unsigned exp_stats_buffer_B_id,
		const unsigned dynamic_launch_flag_buffer_id,
		const unsigned m,
		const unsigned n,
		float* const ptr,
		const unsigned ld,
		const unsigned batch_size = 1,
		const unsigned stride = 0
		);
float get_max_exp(
		const unsigned dynamic_launch_flag_buffer_id
		);
} // namespace hijack_control
} // namespace cumpsgemm
