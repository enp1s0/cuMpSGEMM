#include <cutf/memory.hpp>
#include <cutf/math.hpp>
#include <cumpsgemm/cumpsgemm.hpp>
#include "handle.hpp"

namespace {
constexpr unsigned warp_size = 32;

__global__ void download_exp_stats(
		cumpsgemm::counter_t* const host_total_counter_ptr,
		cumpsgemm::counter_t* const host_lost_counter_ptr,
		const cumpsgemm::counter_t* const dev_total_counter_ptr,
		const cumpsgemm::counter_t* const dev_lost_counter_ptr,
		const unsigned length
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= length) {
		return;
	}
	host_total_counter_ptr [tid] = dev_total_counter_ptr [tid];
	host_lost_counter_ptr[tid] = dev_lost_counter_ptr[tid];
}
} // unnamed namespace

void cumpsgemm::set_exp_stats_params(
		cuMpSGEMM_handle_t handle,
		const float ignore_threshold,
		const float lost_threshold
		) {
	handle->ignore_threshold = ignore_threshold;
	handle->lost_threshold = lost_threshold;
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
			handle->host_lost_counter,
			handle->dev_total_counter,
			handle->dev_lost_counter,
			handle->last_stored_counter_length
			);
	CUTF_CHECK_ERROR(cudaStreamSynchronize(handle->cuda_stream));
	std::vector<std::pair<std::size_t, std::size_t>> result(handle->last_stored_counter_length);

	for (unsigned i = 0; i < handle->last_stored_counter_length; i++) {
		result[i] = std::make_pair<std::size_t, std::size_t>(handle->host_total_counter[i], handle->host_lost_counter[i]);
	}

	return result;
}

void cumpsgemm::exp_stats::resize_counter(
		cuMpSGEMM_handle_t handle,
		const std::size_t new_length
		) {
	CUTF_CHECK_ERROR(cudaFree    (handle->dev_lost_counter ));
	CUTF_CHECK_ERROR(cudaFree    (handle->dev_total_counter  ));
	CUTF_CHECK_ERROR(cudaFreeHost(handle->host_lost_counter));
	CUTF_CHECK_ERROR(cudaFreeHost(handle->host_total_counter ));

	handle->counter_length = new_length;

	CUTF_CHECK_ERROR(cudaMalloc    (&(handle->dev_lost_counter ), sizeof(cumpsgemm::counter_t) * handle->counter_length));
	CUTF_CHECK_ERROR(cudaMalloc    (&(handle->dev_total_counter  ), sizeof(cumpsgemm::counter_t) * handle->counter_length));
	CUTF_CHECK_ERROR(cudaMallocHost(&(handle->host_lost_counter), sizeof(cumpsgemm::counter_t) * handle->counter_length));
	CUTF_CHECK_ERROR(cudaMallocHost(&(handle->host_total_counter ), sizeof(cumpsgemm::counter_t) * handle->counter_length));
}

namespace {
__global__ void init_counter_kernel(
		cumpsgemm::counter_t* const total_counter_ptr,
		cumpsgemm::counter_t* const lost_counter_ptr,
		const unsigned length
		) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= length) {
		return;
	}
	total_counter_ptr[tid] = 0;
	lost_counter_ptr [tid] = 0;
}
} // unnamed namespace

void cumpsgemm::exp_stats::init_counter (
		cuMpSGEMM_handle_t handle,
		const unsigned length
		) {
	const auto block_size = 256u;
	init_counter_kernel<<<(length + block_size - 1) / block_size, block_size, 0, handle->cuda_stream>>>(
			handle->dev_total_counter,
			handle->dev_lost_counter,
			length
			);
#ifdef CUMPSGEMM_CHECK_KERNEL_ERROR
	CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
#endif
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

	local_total_counter = smem_lose_counter_ptr [threadIdx.x];
	local_lose_counter  = smem_total_counter_ptr[threadIdx.x];

	for (std::uint32_t offset = (BLOCK_SIZE / warp_size) >> 1; offset >= 1; offset >>= 1) {
		local_lose_counter  += __shfl_xor_sync(~0u, local_lose_counter , offset);
		local_total_counter += __shfl_xor_sync(~0u, local_total_counter, offset);
	}

	if (threadIdx.x == 0) {
		atomicAdd(lose_counter  + ib, local_lose_counter);
		atomicAdd(total_counter + ib, local_total_counter);
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
	cumpsgemm::exp_stats::init_counter(
			handle,
			batch_size
			);

	constexpr auto block_size = 256;
	const dim3 grid_size(
			std::min<std::uint64_t>(((1lu * m * n) + block_size - 1) / block_size, handle->num_sms * 8),
			batch_size
			);

	exp_stats_ext_kernel<block_size><<<grid_size, block_size, 0, handle->cuda_stream>>>(
			handle->dev_lost_counter,
			handle->dev_total_counter,
			m, n,
			ptr, ld,
			batch_size, stride,
			handle->lost_threshold,
			handle->ignore_threshold
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
