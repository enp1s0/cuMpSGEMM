#pragma once
#include "handle.hpp"

namespace cumpsgemm {
namespace dynamic_scaling {
template <class T>
void scale_AB(
		cuMpSGEMM_handle* handle,
		const unsigned m,
		const unsigned n,
		T* const ptr, const unsigned ld,
		const unsigned stride,
		const unsigned batch_size,
		const unsigned exp_stats_buffer_id,
		const unsigned dynamic_launch_buffer_id
		);
template <class T>
void scale_C(
		cuMpSGEMM_handle* handle,
		const unsigned m,
		const unsigned n,
		T* const ptr, const unsigned ld,
		const unsigned stride,
		const unsigned batch_size,
		const unsigned exp_stats_buffer_A_id,
		const unsigned exp_stats_buffer_B_id,
		const unsigned dynamic_launch_buffer_id
		);
float get_max_exp(
		cuMpSGEMM_handle* handle,
		const unsigned exp_stats_buffer_id
		);
} // dynamic_scaling
} // cumpsgemm
