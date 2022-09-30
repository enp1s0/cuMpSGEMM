#include <cutf/memory.hpp>
#include <cutf/math.hpp>
#include <cumpsgemm/cumpsgemm.hpp>
#include "exp_stats.hpp"

namespace {
constexpr unsigned warp_size = 32;

__global__ void download_exp_stats_kernel(
		cumpsgemm::counter_t* const host_total_counter_ptr,
		cumpsgemm::counter_t* const host_lost_counter_ptr,
		const cumpsgemm::counter_t* const dev_total_counter_ptr,
		const cumpsgemm::counter_t* const dev_lost_counter_ptr
		) {
	*host_total_counter_ptr = *dev_total_counter_ptr;
	*host_lost_counter_ptr = *dev_lost_counter_ptr;
}
} // unnamed namespace

void cumpsgemm::exp_stats::download_exp_stats(
		cuMpSGEMM_handle* handle,
		const unsigned buffer_id
		) {
	download_exp_stats_kernel<<<1, 1, 0, handle->cuda_stream>>>(
			handle->exp_stats_handle->host_total_counter_buffer + buffer_id,
			handle->exp_stats_handle->host_lost_counter_buffer + buffer_id,
			handle->exp_stats_handle->dev_total_counter_buffer + buffer_id,
			handle->exp_stats_handle->dev_lost_counter_buffer + buffer_id
			);
}

std::pair<std::size_t, std::size_t> cumpsgemm::exp_stats::get_exp_stats(
		cuMpSGEMM_handle* handle,
		const unsigned buffer_id
		) {
	while(handle->exp_stats_handle->host_total_counter_buffer[buffer_id] == handle->exp_stats_handle->buffer_empty_value) {}
	while(handle->exp_stats_handle->host_lost_counter_buffer [buffer_id] == handle->exp_stats_handle->buffer_empty_value) {}

	return std::pair<std::size_t, std::size_t>{
		handle->exp_stats_handle->host_lost_counter_buffer[buffer_id],
		handle->exp_stats_handle->host_total_counter_buffer[buffer_id]
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
	CUTF_CHECK_ERROR(cudaFree    (handle->exp_stats_handle->dev_lost_counter_buffer  ));
	CUTF_CHECK_ERROR(cudaFree    (handle->exp_stats_handle->dev_total_counter_buffer ));
	CUTF_CHECK_ERROR(cudaFreeHost(handle->exp_stats_handle->host_lost_counter_buffer ));
	CUTF_CHECK_ERROR(cudaFreeHost(handle->exp_stats_handle->host_total_counter_buffer));

	handle->exp_stats_handle->buffer_length = new_length;

	CUTF_CHECK_ERROR(cudaMalloc    (&(handle->exp_stats_handle->dev_lost_counter_buffer)  , sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
	CUTF_CHECK_ERROR(cudaMalloc    (&(handle->exp_stats_handle->dev_total_counter_buffer) , sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
	CUTF_CHECK_ERROR(cudaMallocHost(&(handle->exp_stats_handle->host_lost_counter_buffer) , sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
	CUTF_CHECK_ERROR(cudaMallocHost(&(handle->exp_stats_handle->host_total_counter_buffer), sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
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
	*(handle->exp_stats_handle->host_total_counter_buffer + buffer_id) = handle->exp_stats_handle->buffer_empty_value;
	*(handle->exp_stats_handle->host_lost_counter_buffer  + buffer_id) = handle->exp_stats_handle->buffer_empty_value;
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
	const auto next = ++(handle->exp_stats_handle->current_buffer_id);
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
template <unsigned BLOCK_SIZE>
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
	for (std::size_t gid = 0; gid <= m * n; gid += blockDim.x * gridDim.x) {
		const auto im = gid % m;
		const auto in = (gid / m) % n;

		const auto memory_index = im + ld * in + stride * ib;

		const auto v = cutf::math::abs(ptr[memory_index]);

		if (v > ignore_threshold) {
			local_total_counter++;
			if (v < lose_threshold) {
				local_lose_counter++;
			}
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

	constexpr auto block_size = 256;
	const dim3 grid_size(
			std::min<std::uint64_t>((1lu * m * n + block_size - 1) / block_size, handle->num_sms * 8),
			batch_size
			);

	exp_stats_ext_kernel<block_size><<<grid_size, block_size, 0, handle->cuda_stream>>>(
			handle->exp_stats_handle->dev_lost_counter_buffer + buffer_id,
			handle->exp_stats_handle->dev_total_counter_buffer + buffer_id,
			m, n,
			ptr, ld,
			batch_size, stride,
			handle->exp_stats_handle->lost_threshold,
			handle->exp_stats_handle->ignore_threshold
			);
	cumpsgemm::exp_stats::download_exp_stats(handle, buffer_id);
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
	CUTF_CHECK_ERROR(cudaMalloc    (&(handle->exp_stats_handle->dev_lost_counter_buffer ) , sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
	CUTF_CHECK_ERROR(cudaMalloc    (&(handle->exp_stats_handle->dev_total_counter_buffer) , sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
	CUTF_CHECK_ERROR(cudaMallocHost(&(handle->exp_stats_handle->host_lost_counter_buffer ), sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
	CUTF_CHECK_ERROR(cudaMallocHost(&(handle->exp_stats_handle->host_total_counter_buffer), sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));

	// For not exp_stats-d matrices
	handle->exp_stats_handle->host_lost_counter_buffer [0] = 1;
	handle->exp_stats_handle->host_total_counter_buffer[0] = 1;
	handle->exp_stats_handle->host_lost_counter_buffer [1] = 0;
	handle->exp_stats_handle->host_total_counter_buffer[1] = 1;
}

void destroy_exp_stats_counter_buffer(
		cuMpSGEMM_handle* handle
		) {
	CUTF_CHECK_ERROR(cudaFree    (handle->exp_stats_handle->dev_lost_counter_buffer  ));
	CUTF_CHECK_ERROR(cudaFree    (handle->exp_stats_handle->dev_total_counter_buffer ));
	CUTF_CHECK_ERROR(cudaFreeHost(handle->exp_stats_handle->host_lost_counter_buffer ));
	CUTF_CHECK_ERROR(cudaFreeHost(handle->exp_stats_handle->host_total_counter_buffer));

	delete handle->exp_stats_handle;
}
