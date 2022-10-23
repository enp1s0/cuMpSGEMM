#include <cutf/memory.hpp>
#include <cutf/math.hpp>
#include <cutf/experimental/fp.hpp>
#include <cumpsgemm/cumpsgemm.hpp>
#include <thread>
#include "exp_stats.hpp"

namespace {
constexpr unsigned warp_size = 32;

__global__ void download_exp_stats_kernel(
		ulong2* const host_result_counter_ptr,
		const cumpsgemm::counter_t* const dev_total_counter_ptr,
		const cumpsgemm::counter_t* const dev_lose_counter_ptr
		) {
	const auto v = make_ulong2(*dev_total_counter_ptr, *dev_lose_counter_ptr);
	*host_result_counter_ptr = v;
}
} // unnamed namespace

void cumpsgemm::exp_stats::download_exp_stats(
		cuMpSGEMM_handle* handle,
		const unsigned buffer_id
		) {
	download_exp_stats_kernel<<<1, 1, 0, handle->cuda_stream>>>(
			handle->exp_stats_handle->host_counter_buffer + buffer_id,
			handle->exp_stats_handle->dev_total_counter_buffer + buffer_id,
			handle->exp_stats_handle->dev_lose_counter_buffer + buffer_id
			);
}

std::pair<std::size_t, std::size_t> cumpsgemm::exp_stats::get_exp_stats(
		cuMpSGEMM_handle* handle,
		const unsigned buffer_id
		) {
	volatile ulong2* p1 = handle->exp_stats_handle->host_counter_buffer;
	while (p1[buffer_id].x == handle->exp_stats_handle->buffer_empty_value) {
	}

	return std::pair<std::size_t, std::size_t>{
		p1[buffer_id].y,
		p1[buffer_id].x
	};
}

void cumpsgemm::set_exp_stats_params(
		cuMpSGEMM_handle_t handle,
		const float ignore_threshold,
		const float lose_threshold
		) {
	handle->exp_stats_handle->ignore_threshold = ignore_threshold;
	handle->exp_stats_handle->lose_threshold = lose_threshold;
}

void cumpsgemm::enable_exp_stats(
		cuMpSGEMM_handle_t handle
		) {
	handle->exp_stats_handle->enabled = true;
}

void cumpsgemm::disable_exp_stats(
		cuMpSGEMM_handle_t handle
		) {
	handle->exp_stats_handle->enabled = false;
}

void cumpsgemm::exp_stats::resize_counter(
		cuMpSGEMM_handle_t handle,
		const std::size_t new_length
		) {
	CUTF_CHECK_ERROR(cudaFree    (handle->exp_stats_handle->dev_lose_counter_buffer ));
	CUTF_CHECK_ERROR(cudaFree    (handle->exp_stats_handle->dev_total_counter_buffer));
	CUTF_CHECK_ERROR(cudaFree    (handle->exp_stats_handle->dev_max_abs_buffer      ));
	CUTF_CHECK_ERROR(cudaFreeHost(handle->exp_stats_handle->host_counter_buffer     ));

	handle->exp_stats_handle->buffer_length = new_length;

	CUTF_CHECK_ERROR(cudaMalloc    (&(handle->exp_stats_handle->dev_lose_counter_buffer ), sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
	CUTF_CHECK_ERROR(cudaMalloc    (&(handle->exp_stats_handle->dev_total_counter_buffer), sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
	CUTF_CHECK_ERROR(cudaMalloc    (&(handle->exp_stats_handle->dev_max_abs_buffer      ), sizeof(float               ) * handle->exp_stats_handle->buffer_length));
	CUTF_CHECK_ERROR(cudaMallocHost(&(handle->exp_stats_handle->host_counter_buffer     ), sizeof(ulong2)               * handle->exp_stats_handle->buffer_length));
}

namespace {
__global__ void init_counter_kernel(
		cumpsgemm::counter_t* const total_counter_ptr,
		cumpsgemm::counter_t* const lose_counter_ptr,
		float* const max_abs_value_ptr
		) {
	*total_counter_ptr = 0;
	*lose_counter_ptr  = 0;
	*max_abs_value_ptr = 0;
}
} // unnamed namespace

void cumpsgemm::exp_stats::init_counter (
		cuMpSGEMM_handle_t handle,
		const unsigned buffer_id
		) {
	init_counter_kernel<<<1, 1, 0, handle->cuda_stream>>>(
			handle->exp_stats_handle->dev_total_counter_buffer + buffer_id,
			handle->exp_stats_handle->dev_lose_counter_buffer + buffer_id,
			handle->exp_stats_handle->dev_max_abs_buffer + buffer_id
			);
	(*(handle->exp_stats_handle->host_counter_buffer + buffer_id)).x = handle->exp_stats_handle->buffer_empty_value;
#ifdef CUMPSGEMM_CHECK_KERNEL_ERROR
	CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
#endif
}

// Ring buffer id calculator.
// 0 and 1 is reserved
// loop[2, 3, ..., buffer_length-1]
std::uint32_t cumpsgemm::exp_stats::get_next_exp_stats_buffer_id(
		cuMpSGEMM_handle* handle
		) {
	handle->exp_stats_handle->current_buffer_id++;
	const auto next = handle->exp_stats_handle->current_buffer_id;
	if (next < handle->exp_stats_handle->buffer_length) {
		return next;
	}
	handle->exp_stats_handle->current_buffer_id = 2;
	return 2;
}
std::uint32_t cumpsgemm::exp_stats::get_current_exp_stats_buffer_id(
		cuMpSGEMM_handle* handle
		) {
	return handle->exp_stats_handle->current_buffer_id;
}

namespace {
__device__ float abs_max_float(const float a) {return cutf::math::abs(a);}
__device__ float abs_max_float(const cuComplex a) {return cutf::math::max(cutf::math::abs(a.x), cutf::math::abs(a.y));}

template <class T>
__device__ T make_zero() {return 0;}
template <>
__device__ cuComplex make_zero() {return make_float2(0, 0);}

// Get the largest abs value
template <unsigned BLOCK_SIZE, unsigned VEC_LEN, class LOOP_T, class T>
__global__ void exp_stats_ext_stage_1_kernel(
		float * const result_ptr,
		const unsigned m,
		const unsigned n,
		const T* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride
		) {
	float local_max_abs_value = 0;
	const auto ib = blockIdx.y;
	const auto local_mat_ptr = ptr + ib * stride;
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
		} else {
			for (uint32_t i = 0; i < VEC_LEN; i++) {
				const auto gid = lid + i;
				T v;
				if (gid < m * n) {
					const auto im = gid % m;
					const auto in = gid / m;

					const auto memory_index = im + ld * in;
					v = local_mat_ptr[memory_index];
				} else {
					v = make_zero<T>();
				}
				vec[i] = v;
			}
		}
		for (uint32_t i = 0; i < VEC_LEN; i++) {
			local_max_abs_value = cutf::math::max(local_max_abs_value, abs_max_float(vec[i]));
		}
	}

	for (std::uint32_t offset = warp_size >> 1; offset >= 1; offset >>= 1) {
		local_max_abs_value = cutf::math::max(
				__shfl_xor_sync(~0u, local_max_abs_value, offset),
				local_max_abs_value
				);
	}

	__shared__ float smem[BLOCK_SIZE];

	if ((threadIdx.x & 0x1f) == 0) {
		smem[threadIdx.x >> 5] = local_max_abs_value;
	}
	__syncthreads();

	if (threadIdx.x >= BLOCK_SIZE / warp_size) return;

	local_max_abs_value = smem[threadIdx.x];

	for (std::uint32_t offset = (BLOCK_SIZE / warp_size) >> 1; offset >= 1; offset >>= 1) {
		local_max_abs_value = cutf::math::max(
				__shfl_xor_sync(~0u, local_max_abs_value, offset),
				local_max_abs_value
				);
	}

	if (threadIdx.x == 0) {
		const std::uint32_t max_abs = cutf::experimental::fp::reinterpret_as_uint(local_max_abs_value) & 0x7fa00000u;
		atomicMax(reinterpret_cast<std::uint32_t*>(result_ptr), max_abs);
	}
}

__device__ void update_counter(
		unsigned& lose_counter,
		unsigned& total_counter,
		const float lose_threshold,
		const float ignore_threshold,
		const float w
		) {
	const auto v = cutf::math::abs(w);
	if (v > ignore_threshold) {
		total_counter++;
		if (v < lose_threshold) {
			lose_counter++;
		}
	}
}

__device__ void update_counter(
		unsigned& lose_counter,
		unsigned& total_counter,
		const float lose_threshold,
		const float ignore_threshold,
		const cuComplex w
		) {
	update_counter(total_counter, lose_counter, ignore_threshold, lose_threshold, w.x);
	update_counter(total_counter, lose_counter, ignore_threshold, lose_threshold, w.y);
}


// exp_stats for cuBLAS original functions
template <unsigned BLOCK_SIZE, unsigned VEC_LEN, class LOOP_T, class T>
__global__ void exp_stats_ext_stage_2_kernel(
		unsigned long long int* const lose_counter,
		unsigned long long int* const total_counter,
		const unsigned m,
		const unsigned n,
		const T* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride,
		const float* const max_abs_ptr,
		const float lose_threshold,
		const float ignore_threshold
		) {
	unsigned local_lose_counter = 0;
	unsigned local_total_counter = 0;
	const auto ib = blockIdx.y;
	const auto local_mat_ptr = ptr + ib * stride;
	const auto max_abs_value = *max_abs_ptr;
	const auto abs_ignore_threshold = ignore_threshold * max_abs_value;
	const auto abs_lose_threshold = lose_threshold * max_abs_value;

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
		} else {
			for (uint32_t i = 0; i < VEC_LEN; i++) {
				const auto gid = lid + i;
				T v;
				if (gid < m * n) {
					const auto im = gid % m;
					const auto in = gid / m;

					const auto memory_index = im + ld * in;
					v = local_mat_ptr[memory_index];
				} else {
					v = make_zero<T>();
				}
				vec[i] = v;
			}
		}

		for (unsigned i = 0; i < VEC_LEN; i++) {
			update_counter(local_lose_counter, local_total_counter, abs_lose_threshold, abs_ignore_threshold, vec[i]);
		}
	}

	for (std::uint32_t offset = warp_size >> 1; offset >= 1; offset >>= 1) {
		local_lose_counter  += __shfl_xor_sync(~0u, local_lose_counter , offset);
		local_total_counter += __shfl_xor_sync(~0u, local_total_counter, offset);
	}

	__shared__ unsigned smem[BLOCK_SIZE];
	unsigned *smem_lose_counter_ptr = reinterpret_cast<unsigned*>(smem);
	unsigned *smem_total_counter_ptr  = smem_lose_counter_ptr + (BLOCK_SIZE / warp_size);

	if ((threadIdx.x & 0x1f) == 0) {
		smem_lose_counter_ptr [threadIdx.x >> 5] = local_lose_counter;
		smem_total_counter_ptr[threadIdx.x >> 5] = local_total_counter;
	}
	__syncthreads();

	if (threadIdx.x >= BLOCK_SIZE / warp_size) return;

	local_total_counter = smem_total_counter_ptr[threadIdx.x];
	local_lose_counter  = smem_lose_counter_ptr [threadIdx.x];

	for (std::uint32_t offset = (BLOCK_SIZE / warp_size) >> 1; offset >= 1; offset >>= 1) {
		local_lose_counter  += __shfl_xor_sync(~0u, local_lose_counter , offset);
		local_total_counter += __shfl_xor_sync(~0u, local_total_counter, offset);
	}

	if (threadIdx.x == 0) {
		atomicAdd(lose_counter , local_lose_counter);
		atomicAdd(total_counter, local_total_counter);
	}
}

template <class T, unsigned BLOCK_SIZE, unsigned VEC_LEN, class LOOP_T>
void launch_exp_stats_ext(
		cuMpSGEMM_handle* handle,
		const unsigned m,
		const unsigned n,
		const T* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride,
		const unsigned buffer_id
		) {
		const dim3 grid_size(
				std::min<std::uint64_t>(((1lu * m * n + BLOCK_SIZE - 1) / BLOCK_SIZE + VEC_LEN - 1) / VEC_LEN, handle->num_sms * 4),
				(stride == 0) ? 1 : batch_size
				);
		exp_stats_ext_stage_1_kernel<BLOCK_SIZE, VEC_LEN, LOOP_T, T><<<grid_size, BLOCK_SIZE, 0, handle->cuda_stream>>>(
				handle->exp_stats_handle->dev_max_abs_buffer + buffer_id,
				m, n,
				ptr, ld,
				batch_size, stride
				);
		exp_stats_ext_stage_2_kernel<BLOCK_SIZE, VEC_LEN, LOOP_T, T><<<grid_size, BLOCK_SIZE, 0, handle->cuda_stream>>>(
				handle->exp_stats_handle->dev_lose_counter_buffer + buffer_id,
				handle->exp_stats_handle->dev_total_counter_buffer + buffer_id,
				m, n,
				ptr, ld,
				batch_size, stride,
				handle->exp_stats_handle->dev_max_abs_buffer + buffer_id,
				handle->exp_stats_handle->lose_threshold,
				handle->exp_stats_handle->ignore_threshold
				);
}
} // unnamed namespace

template <class T>
void cumpsgemm::exp_stats::exp_stats_ext(
		cuMpSGEMM_handle* handle,
		const unsigned m,
		const unsigned n,
		const T* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride
		) {
	const auto buffer_id = cumpsgemm::exp_stats::get_next_exp_stats_buffer_id(handle);
	cumpsgemm::exp_stats::init_counter(
			handle,
			buffer_id
			);
	if (static_cast<std::size_t>(m) * n < (1lu << 15)) {
		launch_exp_stats_ext<T, 64, 4, unsigned>(handle, m, n, ptr, ld, batch_size, stride, buffer_id);
	} else if (static_cast<std::size_t>(m) * n < (1lu << 22)) {
		launch_exp_stats_ext<T, 128, 4, unsigned>(handle, m, n, ptr, ld, batch_size, stride, buffer_id);
	} else if (static_cast<std::size_t>(m) * n < (1lu << 32)) {
		launch_exp_stats_ext<T, 1024, 4, unsigned>(handle, m, n, ptr, ld, batch_size, stride, buffer_id);
	} else {
		launch_exp_stats_ext<T, 1024, 4, std::size_t>(handle, m, n, ptr, ld, batch_size, stride, buffer_id);
	}
}

template void cumpsgemm::exp_stats::exp_stats_ext<float    >(cuMpSGEMM_handle*, const unsigned, const unsigned, const float*     const, const unsigned, const unsigned, const unsigned);
template void cumpsgemm::exp_stats::exp_stats_ext<cuComplex>(cuMpSGEMM_handle*, const unsigned, const unsigned, const cuComplex* const, const unsigned, const unsigned, const unsigned);

void cumpsgemm::exp_stats::reset_exp_stats_buffer_id(
		cuMpSGEMM_handle* handle
		) {
	handle->exp_stats_handle->current_buffer_id = 1;
}

void init_exp_stats_counter_buffer(
		cuMpSGEMM_handle* handle
		) {
	handle->exp_stats_handle = new cumpsgemm::exp_stats::exp_stats_handle;
	cumpsgemm::exp_stats::reset_exp_stats_buffer_id(handle);

	handle->exp_stats_handle->enabled = false;
	handle->exp_stats_handle->buffer_length = 10000;
	handle->exp_stats_handle->ignore_threshold = 0;
	handle->exp_stats_handle->lose_threshold = 0;
	handle->exp_stats_handle->counter_init_disabled = false;
	CUTF_CHECK_ERROR(cudaMalloc    (&(handle->exp_stats_handle->dev_lose_counter_buffer ), sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
	CUTF_CHECK_ERROR(cudaMalloc    (&(handle->exp_stats_handle->dev_total_counter_buffer), sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
	CUTF_CHECK_ERROR(cudaMalloc    (&(handle->exp_stats_handle->dev_max_abs_buffer      ), sizeof(float               ) * handle->exp_stats_handle->buffer_length));
	CUTF_CHECK_ERROR(cudaMallocHost(&(handle->exp_stats_handle->host_counter_buffer     ), sizeof(ulong2              ) * handle->exp_stats_handle->buffer_length));

	// For not exp_stats-d matrices
	handle->exp_stats_handle->host_counter_buffer[0].x = 1;
	handle->exp_stats_handle->host_counter_buffer[0].y = 1;
	handle->exp_stats_handle->host_counter_buffer[1].x = 0;
	handle->exp_stats_handle->host_counter_buffer[1].y = 1;
}

void destroy_exp_stats_counter_buffer(
		cuMpSGEMM_handle* handle
		) {
	CUTF_CHECK_ERROR(cudaFree    (handle->exp_stats_handle->dev_lose_counter_buffer ));
	CUTF_CHECK_ERROR(cudaFree    (handle->exp_stats_handle->dev_total_counter_buffer));
	CUTF_CHECK_ERROR(cudaFreeHost(handle->exp_stats_handle->host_counter_buffer     ));

	delete handle->exp_stats_handle;
}
