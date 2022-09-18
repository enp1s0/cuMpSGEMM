#include <cutf/memory.hpp>
#include <cumpsgemm/cumpsgemm.hpp>
#include "handle.hpp"

namespace {
__global__ void download_exp_stats(
		cumpsgemm::counter_t* const host_total_counter_ptr,
		cumpsgemm::counter_t* const host_target_counter_ptr,
		const cumpsgemm::counter_t* const dev_total_counter_ptr,
		const cumpsgemm::counter_t* const dev_target_counter_ptr,
		const unsigned length
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= length) {
		return;
	}
	host_total_counter_ptr [tid] = dev_total_counter_ptr [tid];
	host_target_counter_ptr[tid] = dev_target_counter_ptr[tid];
}
} // unnamed namespace

void cumpsgemm::set_exp_stats_params(
		cuMpSGEMM_handle_t handle,
		const float ignore_threshold,
		const float target_threshold
		) {
	handle->ignore_threshold = ignore_threshold;
	handle->target_threshold = target_threshold;
}

void cumpsgemm::enable_exp_stats(
		cuMpSGEMM_handle_t handle
		) {
	handle->exp_stats_enabled = true;
}

void cumpsgemm::disable_exp_stats(
		cuMpSGEMM_handle_t handle
		) {
	handle->exp_stats_enabled = false;
}

std::vector<std::pair<std::size_t, std::size_t>> cumpsgemm::get_last_exp_stats(
		cuMpSGEMM_handle_t handle
		) {
	const auto block_size = 256u;
	download_exp_stats<<<(handle->last_stored_counter_length + block_size - 1) / block_size, block_size, 0, handle->cuda_stream>>>(
			handle->host_total_counter,
			handle->host_target_counter,
			handle->dev_total_counter,
			handle->dev_target_counter,
			handle->last_stored_counter_length
			);
	CUTF_CHECK_ERROR(cudaStreamSynchronize(handle->cuda_stream));
	std::vector<std::pair<std::size_t, std::size_t>> result(handle->last_stored_counter_length);

	for (unsigned i = 0; i < handle->last_stored_counter_length; i++) {
		result[i] = std::make_pair<std::size_t, std::size_t>(handle->host_total_counter[i], handle->host_target_counter[i]);
	}

	return result;
}
