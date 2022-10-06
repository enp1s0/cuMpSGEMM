#include <cutf/memory.hpp>
#include <cumpsgemm/hijack_control.hpp>
#include "dynamic_launch.hpp"

namespace {
__global__ void init_flag_buffer(
		int* const ptr
		) {
	ptr[0] = CUMPSGEMM_TF32TCEC;
	ptr[1] = CUMPSGEMM_FP16TCEC;
}
} // unnamed namespace

void init_dynamic_launch_flag_buffer(
		cuMpSGEMM_handle* handle
		) {
	handle->dynamic_launch_handle = new cumpsgemm::dynamic_launch::dynamic_launch_handle;

	handle->dynamic_launch_handle->flag_buffer_length = 10000;
	handle->dynamic_launch_handle->current_buffer_id = 1;
	handle->dynamic_launch_handle->enabled = false;
	handle->dynamic_launch_handle->enabled_id = 0;

	handle->dynamic_launch_handle->mode_A = CUMPSGEMM_TF32TCEC;
	handle->dynamic_launch_handle->mode_B = CUMPSGEMM_FP16TCEC;

	CUTF_CHECK_ERROR(cudaMalloc(&handle->dynamic_launch_handle->frag_buffer, sizeof(int) * handle->dynamic_launch_handle->flag_buffer_length));
	init_flag_buffer<<<1, 1, 0, handle->cuda_stream>>>(handle->dynamic_launch_handle->frag_buffer);
}

void destroy_launch_flag_buffer(
		cuMpSGEMM_handle* handle
		) {
	CUTF_CHECK_ERROR(cudaFree(handle->dynamic_launch_handle->frag_buffer));
	delete handle->dynamic_launch_handle;
}

unsigned cumpsgemm::dynamic_launch::get_next_dynamic_launch_flag_buffer_id(
		cuMpSGEMM_handle* handle
		) {
	const auto next = ++(handle->dynamic_launch_handle->current_buffer_id);
	if (next < handle->dynamic_launch_handle->flag_buffer_length) {
		return next;
	}
	handle->dynamic_launch_handle->flag_buffer_length = 2;
	return 2;
}

void cumpsgemm::dynamic_launch::set_dynamic_launch_flag_buffer_id(
		cuMpSGEMM_handle* handle,
		const unsigned id
		) {
	handle->dynamic_launch_handle->enabled_id = id;
	handle->dynamic_launch_handle->enabled = 1;
}

void cumpsgemm::dynamic_launch::unset_dynamic_launch_flag_buffer_id(
		cuMpSGEMM_handle* handle
		) {
	handle->dynamic_launch_handle->enabled_id = 0;
	handle->dynamic_launch_handle->enabled = 0;
}
