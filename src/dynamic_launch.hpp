#pragma once
#include <cstdint>
#include <cumpsgemm/detail/common.h>
#include "handle.hpp"

namespace cumpsgemm {
namespace dynamic_launch {
struct dynamic_launch_handle {
	unsigned flag_buffer_length;
	unsigned current_buffer_id;

	int *frag_buffer;

	bool enabled;
	unsigned enabled_id;

	cuMpSGEMM_compute_mode_t mode_A;
	cuMpSGEMM_compute_mode_t mode_B;
};

unsigned get_next_dynamic_launch_flag_buffer_id(cuMpSGEMM_handle* handle);
void set_dynamic_launch_flag_buffer_id(cuMpSGEMM_handle* handle, unsigned id);
void unset_dynamic_launch_flag_buffer_id(cuMpSGEMM_handle* handle);
void set_compute_mode_AB(cuMpSGEMM_handle* handle, const cuMpSGEMM_compute_mode_t mode_A, const cuMpSGEMM_compute_mode_t mode_B);
} // namespace dynamic_launch
} // namespace cumpsgemm
