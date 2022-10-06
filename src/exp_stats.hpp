#pragma once
#include "handle.hpp"

namespace cumpsgemm {
namespace exp_stats {
struct exp_stats_handle {
	bool enabled;

	cumpsgemm::counter_t* dev_total_counter_buffer;
	cumpsgemm::counter_t* dev_lose_counter_buffer;
	ulong2* host_counter_buffer;
	static constexpr cumpsgemm::counter_t buffer_empty_value = ~0llu;

	float ignore_threshold;
	float lose_threshold;

	std::uint32_t buffer_length;
	std::uint32_t current_buffer_id;
	bool counter_init_disabled;
};
// exp_stats API
void resize_counter(
		cuMpSGEMM_handle* handle,
		const std::size_t new_length
		);
void init_counter (
		cuMpSGEMM_handle* handle,
		const unsigned buffer_id
		);
std::uint32_t get_next_exp_stats_buffer_id(
		cuMpSGEMM_handle* handle
		);
std::uint32_t get_current_exp_stats_buffer_id(
		cuMpSGEMM_handle* handle
		);
void reset_exp_stats_buffer_id(
		cuMpSGEMM_handle* handle
		);
void download_exp_stats(
		cuMpSGEMM_handle* handle,
		const unsigned buffer_id
		);
std::pair<std::size_t, std::size_t> get_exp_stats(
		cuMpSGEMM_handle* handle,
		const unsigned buffer_id
		);
void exp_stats_ext(
		cuMpSGEMM_handle* handle,
		const unsigned m,
		const unsigned n,
		const float* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride
		);
void exp_stats_ext(
		cuMpSGEMM_handle* handle,
		const unsigned m,
		const unsigned n,
		const cuComplex* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride
		);
} // namespace exp_stats
} // namespace cumpsgemm
