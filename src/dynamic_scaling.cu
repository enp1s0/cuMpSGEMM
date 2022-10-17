#include <algorithm>
#include <cumpsgemm/detail/common.h>
#include <cutf/memory.hpp>
#include "dynamic_scaling.hpp"
#include "exp_stats.hpp"
#include "dynamic_launch.hpp"

namespace {
__device__ float  mul_a(const float v, const float a) {return v * a;}
__device__ float2 mul_a(const cuComplex v, const float a) {return make_float2(v.x * a, v.y * a);}
template <unsigned BLOCK_SIZE, unsigned VEC_LEN, class LOOP_T, class T>
__device__ void scaling_core (
		const unsigned m,
		const unsigned n,
		T* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride,
		const float coef
			) {
	const auto ib = blockIdx.y;
	auto local_mat_ptr = ptr + ib * stride;
	for (LOOP_T lid = (threadIdx.x + blockIdx.x * blockDim.x) * VEC_LEN; lid < m * n; lid += BLOCK_SIZE * gridDim.x * VEC_LEN) {
		T vec[VEC_LEN];
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
				local_mat_ptr[memory_index] = mul_a(vec[i], coef);
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
					local_mat_ptr[memory_index] = mul_a(vec[i], coef);
				} else {
					break;
				}
			}
		}
	}
}

template <unsigned BLOCK_SIZE, unsigned VEC_LEN, class T, class LOOP_T>
__global__ void scaling_kernel(
		const int* const dynamic_mode,
		const unsigned m,
		const unsigned n,
		T* const ptr,
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

	scaling_core<BLOCK_SIZE, VEC_LEN, LOOP_T, T>(
			m, n,
			ptr, ld,
			batch_size, stride,
			coef
			);
}

template <unsigned BLOCK_SIZE, unsigned VEC_LEN, class T, class LOOP_T>
__global__ void scaling_kernel(
		const int* const dynamic_mode,
		const unsigned m,
		const unsigned n,
		T* const ptr,
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
	const auto coef = (*max_abs_value_A_ptr * *max_abs_value_B_ptr) / (1u << 30);

	scaling_core<BLOCK_SIZE, VEC_LEN, LOOP_T, T>(
			m, n,
			ptr, ld,
			batch_size, stride,
			coef
			);
}
} // unnamed namespace

template <class T>
void cumpsgemm::dynamic_scaling::scale_AB(
		cuMpSGEMM_handle* handle,
		const unsigned m,
		const unsigned n,
		T* const ptr, const unsigned ld,
		const unsigned stride,
		const unsigned batch_size,
		const unsigned exp_stats_buffer_id,
		const unsigned dynamic_launch_buffer_id
		) {
	constexpr unsigned VEC_LEN = 2;

	constexpr auto block_size = 256;
	const dim3 grid_size(
			((1lu * m * n + block_size - 1) / block_size + VEC_LEN - 1) / VEC_LEN,
			batch_size
			);

	if (static_cast<std::size_t>(m) * n < (1lu << 32)) {
		using LOOP_T = unsigned;
		scaling_kernel<block_size, VEC_LEN, T, LOOP_T><<<grid_size, block_size, 0, handle->cuda_stream>>>(
				handle->dynamic_launch_handle->flag_buffer + dynamic_launch_buffer_id,
				m, n,
				ptr, ld,
				batch_size, stride,
				handle->exp_stats_handle->dev_max_abs_buffer + exp_stats_buffer_id
				);
	} else {
		using LOOP_T = std::size_t;
		scaling_kernel<block_size, VEC_LEN, T, LOOP_T><<<grid_size, block_size, 0, handle->cuda_stream>>>(
				handle->dynamic_launch_handle->flag_buffer + dynamic_launch_buffer_id,
				m, n,
				ptr, ld,
				batch_size, stride,
				handle->exp_stats_handle->dev_max_abs_buffer + exp_stats_buffer_id
				);
	}
}
template void cumpsgemm::dynamic_scaling::scale_AB<float>(
		cuMpSGEMM_handle*,
		const unsigned,
		const unsigned,
		float* const, const unsigned,
		const unsigned,
		const unsigned,
		const unsigned,
		const unsigned);
template void cumpsgemm::dynamic_scaling::scale_AB<cuComplex>(
		cuMpSGEMM_handle*,
		const unsigned,
		const unsigned,
		cuComplex* const, const unsigned,
		const unsigned,
		const unsigned,
		const unsigned,
		const unsigned);

template <class T>
void cumpsgemm::dynamic_scaling::scale_C(
		cuMpSGEMM_handle* handle,
		const unsigned m,
		const unsigned n,
		T* const ptr, const unsigned ld,
		const unsigned stride,
		const unsigned batch_size,
		const unsigned exp_stats_buffer_A_id,
		const unsigned exp_stats_buffer_B_id,
		const unsigned dynamic_launch_buffer_id
		) {
	constexpr unsigned VEC_LEN = 2;

	constexpr auto block_size = 256;
	const dim3 grid_size(
			((1lu * m * n + block_size - 1) / block_size + VEC_LEN - 1) / VEC_LEN,
			batch_size
			);

	if (static_cast<std::size_t>(m) * n < (1lu << 32)) {
		using LOOP_T = unsigned;
		scaling_kernel<block_size, VEC_LEN, T, LOOP_T><<<grid_size, block_size, 0, handle->cuda_stream>>>(
				handle->dynamic_launch_handle->flag_buffer + dynamic_launch_buffer_id,
				m, n,
				ptr, ld,
				batch_size, stride,
				handle->exp_stats_handle->dev_max_abs_buffer + exp_stats_buffer_A_id,
				handle->exp_stats_handle->dev_max_abs_buffer + exp_stats_buffer_B_id
				);
	} else {
		using LOOP_T = std::size_t;
		scaling_kernel<block_size, VEC_LEN, T, LOOP_T><<<grid_size, block_size, 0, handle->cuda_stream>>>(
				handle->dynamic_launch_handle->flag_buffer + dynamic_launch_buffer_id,
				m, n,
				ptr, ld,
				batch_size, stride,
				handle->exp_stats_handle->dev_max_abs_buffer + exp_stats_buffer_A_id,
				handle->exp_stats_handle->dev_max_abs_buffer + exp_stats_buffer_B_id
				);
	}
}
template void cumpsgemm::dynamic_scaling::scale_C<float>(
		cuMpSGEMM_handle*,
		const unsigned,
		const unsigned,
		float* const, const unsigned,
		const unsigned,
		const unsigned,
		const unsigned,
		const unsigned,
		const unsigned);
template void cumpsgemm::dynamic_scaling::scale_C<cuComplex>(
		cuMpSGEMM_handle*,
		const unsigned,
		const unsigned,
		cuComplex* const, const unsigned,
		const unsigned,
		const unsigned,
		const unsigned,
		const unsigned,
		const unsigned);

float cumpsgemm::dynamic_scaling::get_max_exp(
		cuMpSGEMM_handle* handle,
		const unsigned exp_stats_buffer_id
		) {
	float max_exp;
	CUTF_CHECK_ERROR(cudaMemcpyAsync(
				&max_exp,
				handle->exp_stats_handle->dev_max_abs_buffer + exp_stats_buffer_id
				, sizeof(float),
				cudaMemcpyDefault,
				handle->cuda_stream));
	CUTF_CHECK_ERROR(cudaStreamSynchronize(handle->cuda_stream));
	return max_exp;
}
