#include <cutf/memory.hpp>
#include <cutf/math.hpp>
#include <cumpsgemm/cumpsgemm.hpp>
#include <chrono>
#include <thread>
#include "exp_stats.hpp"

namespace {
constexpr unsigned warp_size = 32;

__global__ void download_exp_stats_kernel(
		ulong2* const host_result_counter_ptr,
		const cumpsgemm::counter_t* const dev_total_counter_ptr,
		const cumpsgemm::counter_t* const dev_lost_counter_ptr
		) {
	const auto v = make_ulong2(*dev_total_counter_ptr, *dev_lost_counter_ptr);
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
			handle->exp_stats_handle->dev_lost_counter_buffer + buffer_id
			);
}

std::pair<std::size_t, std::size_t> cumpsgemm::exp_stats::get_exp_stats(
		cuMpSGEMM_handle* handle,
		const unsigned buffer_id
		) {
	const auto start_clock = std::chrono::system_clock::now();
	volatile ulong2* p1 = handle->exp_stats_handle->host_counter_buffer;
	while (p1[buffer_id].x == handle->exp_stats_handle->buffer_empty_value) {
	}
	const auto end_clock = std::chrono::system_clock::now();

	return std::pair<std::size_t, std::size_t>{
		p1[buffer_id].y,
		p1[buffer_id].x
	};
}

void cumpsgemm::set_exp_stats_params(
		cuMpSGEMM_handle_t handle,
		const float ignore_threshold,
		const float lost_threshold
		) {
	handle->exp_stats_handle->ignore_threshold = ignore_threshold;
	handle->exp_stats_handle->lost_threshold = lost_threshold;
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
	CUTF_CHECK_ERROR(cudaFree    (handle->exp_stats_handle->dev_lost_counter_buffer ));
	CUTF_CHECK_ERROR(cudaFree    (handle->exp_stats_handle->dev_total_counter_buffer));
	CUTF_CHECK_ERROR(cudaFreeHost(handle->exp_stats_handle->host_counter_buffer     ));

	handle->exp_stats_handle->buffer_length = new_length;

	CUTF_CHECK_ERROR(cudaMalloc    (&(handle->exp_stats_handle->dev_lost_counter_buffer) , sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
	CUTF_CHECK_ERROR(cudaMalloc    (&(handle->exp_stats_handle->dev_total_counter_buffer), sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
	CUTF_CHECK_ERROR(cudaMallocHost(&(handle->exp_stats_handle->host_counter_buffer     ), sizeof(ulong2)               * handle->exp_stats_handle->buffer_length));
}

namespace {
__global__ void init_counter_kernel(
		cumpsgemm::counter_t* const total_counter_ptr,
		cumpsgemm::counter_t* const lost_counter_ptr
		) {
	*total_counter_ptr = 0;
	*lost_counter_ptr  = 0;
}
} // unnamed namespace

void cumpsgemm::exp_stats::init_counter (
		cuMpSGEMM_handle_t handle,
		const unsigned buffer_id
		) {
	init_counter_kernel<<<1, 1, 0, handle->cuda_stream>>>(
			handle->exp_stats_handle->dev_total_counter_buffer + buffer_id,
			handle->exp_stats_handle->dev_lost_counter_buffer + buffer_id
			);
	(*(handle->exp_stats_handle->host_counter_buffer + buffer_id)).x = handle->exp_stats_handle->buffer_empty_value;
#ifdef CUMPSGEMM_CHECK_KERNEL_ERROR
	CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
#endif
}

// Ring buffer id calculator.
// 0 and 1 is reserved
// loop[2, 3, ..., buffer_length-1]
std::uint32_t cumpsgemm::exp_stats::get_next_buffer_id(
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
std::uint32_t cumpsgemm::exp_stats::get_current_buffer_id(
		cuMpSGEMM_handle* handle
		) {
	return handle->exp_stats_handle->current_buffer_id;
}

namespace {
// exp_stats for cuBLAS original functions
template <unsigned BLOCK_SIZE, unsigned VEC_LEN>
__global__ void exp_stats_ext_kernel(
		unsigned long long int* const lose_counter,
		unsigned long long int* const total_counter,
		const unsigned m,
		const unsigned n,
		const float* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride,
		const float lose_threshold,
		const float ignore_threshold
		) {
	unsigned local_lose_counter = 0;
	unsigned local_total_counter = 0;
	const auto ib = blockIdx.y;
	const auto local_mat_ptr = ptr + ib * stride;
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
		} else {
			for (uint32_t i = 0; i < VEC_LEN; i++) {
				const auto gid = lid + i;
				float v;
				if (gid < m * n) {
					const auto im = gid % m;
					const auto in = gid / m;

					const auto memory_index = im + ld * in;
					v = local_mat_ptr[memory_index];
				} else {
					v = 0;
				}
				vec[i] = v;
			}
		}

		for (unsigned i = 0; i < VEC_LEN; i++) {
			const auto v = vec[i];
			if (v > ignore_threshold) {
				local_total_counter++;
				if (v < lose_threshold) {
					local_lose_counter++;
				}
			}
		}
		break;
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
} // unnamed namespace

void cumpsgemm::exp_stats::exp_stats_ext(
		cuMpSGEMM_handle* handle,
		const unsigned m,
		const unsigned n,
		const float* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride
		) {
	const auto buffer_id = cumpsgemm::exp_stats::get_next_buffer_id(handle);
	cumpsgemm::exp_stats::init_counter(
			handle,
			buffer_id
			);

	constexpr unsigned VEC_LEN = 8;

	constexpr auto block_size = 1024;
	const dim3 grid_size(
			std::min<std::uint64_t>(((1lu * m * n + block_size - 1) / block_size + VEC_LEN - 1) / VEC_LEN, handle->num_sms),
			batch_size
			);

	exp_stats_ext_kernel<block_size, VEC_LEN><<<grid_size, block_size, 0, handle->cuda_stream>>>(
			handle->exp_stats_handle->dev_lost_counter_buffer + buffer_id,
			handle->exp_stats_handle->dev_total_counter_buffer + buffer_id,
			m, n,
			ptr, ld,
			batch_size, stride,
			handle->exp_stats_handle->lost_threshold,
			handle->exp_stats_handle->ignore_threshold
			);
}

void cumpsgemm::exp_stats::exp_stats_ext(
		cuMpSGEMM_handle* handle,
		const unsigned m,
		const unsigned n,
		const cuComplex* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride
		) {
	cumpsgemm::exp_stats::exp_stats_ext(
			handle,
			2 * m,
			n,
			reinterpret_cast<const float*>(ptr),
			ld,
			batch_size,
			2 * stride
			);
}

void cumpsgemm::exp_stats::reset_buffer_id(
		cuMpSGEMM_handle* handle
		) {
	handle->exp_stats_handle->current_buffer_id = 1;
}

void init_exp_stats_counter_buffer(
		cuMpSGEMM_handle* handle
		) {
	handle->exp_stats_handle = new cumpsgemm::exp_stats::exp_stats_handle;
	cumpsgemm::exp_stats::reset_buffer_id(handle);

	handle->exp_stats_handle->enabled = false;
	handle->exp_stats_handle->buffer_length = 10000;
	handle->exp_stats_handle->ignore_threshold = 0;
	handle->exp_stats_handle->lost_threshold = 0;
	handle->exp_stats_handle->counter_init_disabled = false;
	CUTF_CHECK_ERROR(cudaMalloc    (&(handle->exp_stats_handle->dev_lost_counter_buffer ), sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
	CUTF_CHECK_ERROR(cudaMalloc    (&(handle->exp_stats_handle->dev_total_counter_buffer), sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
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
	CUTF_CHECK_ERROR(cudaFree    (handle->exp_stats_handle->dev_lost_counter_buffer ));
	CUTF_CHECK_ERROR(cudaFree    (handle->exp_stats_handle->dev_total_counter_buffer));
	CUTF_CHECK_ERROR(cudaFreeHost(handle->exp_stats_handle->host_counter_buffer     ));

	delete handle->exp_stats_handle;
}
