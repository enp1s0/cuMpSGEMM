#include <algorithm>
#include <cumpsgemm/detail/common.h>
#include <cutf/memory.hpp>
#include "dynamic_scaling.hpp"
#include "exp_stats.hpp"
#include "dynamic_launch.hpp"

namespace {
template <unsigned BLOCK_SIZE, unsigned VEC_LEN>
__device__ void scaling_core (
		const unsigned m,
		const unsigned n,
		float* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride,
		const float coef
			) {
	const auto ib = blockIdx.y;
	auto local_mat_ptr = ptr + ib * stride;
	for (std::size_t lid = (threadIdx.x + blockIdx.x * blockDim.x) * VEC_LEN; lid < m * n; lid += BLOCK_SIZE * gridDim.x * VEC_LEN) {
		float vec[VEC_LEN];
		if (lid + VEC_LEN < m * n) {
			for (uint32_t i = 0; i < VEC_LEN; i++) {
				const auto gid = lid + i;
				const auto im = gid % m;
				const auto in = gid / m;

				const auto memory_index = im + ld * in;
				vec[i] = local_mat_ptr[memory_index];
			}

			for (uint32_t i = 0; i < VEC_LEN; i++) {
				const auto gid = lid + i;
				const auto im = gid % m;
				const auto in = gid / m;

				const auto memory_index = im + ld * in;
				local_mat_ptr[memory_index] = vec[i] * coef;
			}
		} else {
			for (uint32_t i = 0; i < VEC_LEN; i++) {
				const auto gid = lid + i;
				if (gid < m * n) {
					const auto im = gid % m;
					const auto in = gid / m;

					const auto memory_index = im + ld * in;
					vec[i] = local_mat_ptr[memory_index];
				} else {
					break;
				}
			}
			for (uint32_t i = 0; i < VEC_LEN; i++) {
				const auto gid = lid + i;
				if (gid < m * n) {
					const auto im = gid % m;
					const auto in = gid / m;

					const auto memory_index = im + ld * in;
					local_mat_ptr[memory_index] = vec[i] * coef;
				} else {
					break;
				}
			}
		}
	}
}

template <unsigned BLOCK_SIZE, unsigned VEC_LEN>
__global__ void scaling_kernel(
		const int* const dynamic_mode,
		const unsigned m,
		const unsigned n,
		float* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride,
		const float* const max_abs_value_ptr
		) {
	if (dynamic_mode != nullptr) {
		const auto mode = *dynamic_mode;
		if (mode != CUMPSGEMM_TF32TCEC) return;
	}
	const auto coef = (1u << 15) / *max_abs_value_ptr;

	scaling_core<BLOCK_SIZE, VEC_LEN>(
			m, n,
			ptr, ld,
			batch_size, stride,
			coef
			);
}

template <unsigned BLOCK_SIZE, unsigned VEC_LEN>
__global__ void scaling_kernel(
		const int* const dynamic_mode,
		const unsigned m,
		const unsigned n,
		float* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride,
		const float* const max_abs_value_A_ptr,
		const float* const max_abs_value_B_ptr
		) {
	if (dynamic_mode != nullptr) {
		const auto mode = *dynamic_mode;
		if (mode != CUMPSGEMM_TF32TCEC) return;
	}
	const auto coef = (*max_abs_value_A_ptr * *max_abs_value_B_ptr) / (1u << 15);

	scaling_core<BLOCK_SIZE, VEC_LEN>(
			m, n,
			ptr, ld,
			batch_size, stride,
			coef
			);
}
} // unnamed namespace

void cumpsgemm::dynamic_scaling::scale_AB(
		cuMpSGEMM_handle* handle,
		const unsigned m,
		const unsigned n,
		float* const ptr, const unsigned ld,
		const unsigned stride,
		const unsigned batch_size,
		const unsigned exp_stats_buffer_id,
		const unsigned dynamic_launch_buffer_id
		) {
	constexpr unsigned VEC_LEN = 8;

	constexpr auto block_size = 1024;
	const dim3 grid_size(
			std::min<std::uint64_t>(((1lu * m * n + block_size - 1) / block_size + VEC_LEN - 1) / VEC_LEN, handle->num_sms),
			batch_size
			);

	scaling_kernel<block_size, VEC_LEN><<<grid_size, block_size, 0, handle->cuda_stream>>>(
			handle->dynamic_launch_handle->flag_buffer + dynamic_launch_buffer_id,
			m, n,
			ptr, ld,
			batch_size, stride,
			handle->exp_stats_handle->dev_max_abs_buffer + exp_stats_buffer_id
			);
}

void cumpsgemm::dynamic_scaling::scale_C(
		cuMpSGEMM_handle* handle,
		const unsigned m,
		const unsigned n,
		float* const ptr, const unsigned ld,
		const unsigned stride,
		const unsigned batch_size,
		const unsigned exp_stats_buffer_A_id,
		const unsigned exp_stats_buffer_B_id,
		const unsigned dynamic_launch_buffer_id
		) {
	constexpr unsigned VEC_LEN = 8;

	constexpr auto block_size = 1024;
	const dim3 grid_size(
			std::min<std::uint64_t>(((1lu * m * n + block_size - 1) / block_size + VEC_LEN - 1) / VEC_LEN, handle->num_sms),
			batch_size
			);

	scaling_kernel<block_size, VEC_LEN><<<grid_size, block_size, 0, handle->cuda_stream>>>(
			handle->dynamic_launch_handle->flag_buffer + dynamic_launch_buffer_id,
			m, n,
			ptr, ld,
			batch_size, stride,
			handle->exp_stats_handle->dev_max_abs_buffer + exp_stats_buffer_A_id,
			handle->exp_stats_handle->dev_max_abs_buffer + exp_stats_buffer_B_id
			);
}

float cumpsgemm::dynamic_scaling::get_max_exp(
		cuMpSGEMM_handle* handle,
		const unsigned exp_stats_buffer_id
		) {
	float max_exp;
	CUTF_CHECK_ERROR(cutf::memory::copy_async(&max_exp, handle->exp_stats_handle->dev_max_abs_buffer, sizeof(float), handle->cuda_stream));
	CUTF_CHECK_ERROR(cudaStreamSynchronize(handle->cuda_stream));
	return max_exp;
}
