#ifndef __CUMPGEMM_KERNEL_CUH__
#define __CUMPGEMM_KERNEL_CUH__
#include <cassert>
#include <type_traits>
#include <wmma_extension/utils.hpp>
#include <cutf/cuda.hpp>
#include <cutf/error.hpp>
#include <cutf/cp_async.hpp>

#include <cumpsgemm/cumpsgemm.h>
#include <cumpsgemm/cumpsgemm.hpp>
#include "device_tcec_wrapper.hpp"
#include "handle.hpp"
#include "dmem_accessor.hpp"

namespace {
constexpr unsigned smem_A_skew = 8;
constexpr unsigned smem_B_skew = 8;
constexpr unsigned smem_C_skew = 4;
constexpr unsigned warp_size = 32;

// smem size
template <unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, class Layout>
struct get_smem_size                                             {static constexpr unsigned value = (SMEM_N + SKEW) * SMEM_M;};
template <unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW>
struct get_smem_size<SMEM_M, SMEM_N, SKEW, cumpsgemm::col_major> {static constexpr unsigned value = (SMEM_M + SKEW) * SMEM_N;};

// leading dimension
template <unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, class Layout>
struct get_smem_ld                                             {static constexpr unsigned value = SMEM_M + SKEW;};
template <unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW>
struct get_smem_ld<SMEM_M, SMEM_N, SKEW, cumpsgemm::row_major> {static constexpr unsigned value = SMEM_N + SKEW;};

// smem index
template <unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, class Layout>
struct get_smem_index                                             {__device__ unsigned operator() (const unsigned m, const unsigned n) {return (m + n * (SMEM_M + SKEW));}};
template <unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW>
struct get_smem_index<SMEM_M, SMEM_N, SKEW, cumpsgemm::row_major> {__device__ unsigned operator() (const unsigned m, const unsigned n) {return (n + m * (SMEM_N + SKEW));}};

template <
	class T,
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned BLOCK_SIZE,
	class OP_A,
	class OP_B,
	class TC_T,
	class EC
>
struct mma_smem {
__device__ void operator() (
		cumpsgemm::device::tc_fragment<T, nvcuda::wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, void, TC_T, EC> frag_c[SMEM_M * SMEM_N / (FRAG_M * FRAG_N) / (BLOCK_SIZE / warp_size)],
		const T* const a_smem_ptr,
		const T* const b_smem_ptr
		) {
	static_assert((SMEM_M / FRAG_M) * (SMEM_N / FRAG_N) >= (BLOCK_SIZE / warp_size));
	for (unsigned j = 0; j < (SMEM_M / FRAG_M) * (SMEM_N / FRAG_N); j += BLOCK_SIZE / warp_size) {
		const auto i = j + threadIdx.x / warp_size;
		const unsigned bm = i % (SMEM_M / FRAG_M);
		const unsigned bn = i / (SMEM_M / FRAG_M);

		for (unsigned k = 0; k < SMEM_K; k += FRAG_K) {
			cumpsgemm::device::tc_fragment<T, nvcuda::wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, OP_A, TC_T, EC> frag_a;
			cumpsgemm::device::load_matrix(frag_a, a_smem_ptr + get_smem_index<SMEM_M, SMEM_K, smem_A_skew, OP_A>{}(bm * FRAG_M, k), get_smem_ld<SMEM_M, SMEM_K, smem_A_skew, OP_A>::value);

			cumpsgemm::device::tc_fragment<T, nvcuda::wmma::matrix_b, FRAG_M, FRAG_N, FRAG_K, OP_B, TC_T, EC> frag_b;
			cumpsgemm::device::load_matrix(frag_b, b_smem_ptr + get_smem_index<SMEM_K, SMEM_N, smem_B_skew, OP_B>{}(k, bn * FRAG_N), get_smem_ld<SMEM_K, SMEM_N, smem_B_skew, OP_B>::value);

			cumpsgemm::device::mma(frag_c[i / (BLOCK_SIZE / warp_size)], frag_a, frag_b, frag_c[i / (BLOCK_SIZE / warp_size)]);
		}
	}
}
};

template <
	class T,
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned BLOCK_SIZE,
	class OP_A,
	class OP_B,
	class TC_T,
	class EC
>
struct mma_smem_pipeline {
__device__ void operator() (
		cumpsgemm::device::tc_fragment<T, nvcuda::wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, void, TC_T, EC> frag_c[SMEM_M * SMEM_N / (FRAG_M * FRAG_N) / (BLOCK_SIZE / warp_size)],
		const T* const a_smem_ptr,
		const T* const b_smem_ptr
		) {
	static_assert((SMEM_M / FRAG_M) * (SMEM_N / FRAG_N) >= (BLOCK_SIZE / warp_size));
	for (unsigned j = 0; j < (SMEM_M / FRAG_M) * (SMEM_N / FRAG_N); j += BLOCK_SIZE / warp_size) {
		const auto i = j + threadIdx.x / warp_size;
		const unsigned bm = i % (SMEM_M / FRAG_M);
		const unsigned bn = i / (SMEM_M / FRAG_M);
		cumpsgemm::device::tc_fragment<T, nvcuda::wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, OP_A, TC_T, EC> frag_a[2];
		cumpsgemm::device::tc_fragment<T, nvcuda::wmma::matrix_b, FRAG_M, FRAG_N, FRAG_K, OP_B, TC_T, EC> frag_b[2];

		cumpsgemm::device::load_matrix(frag_a[0], a_smem_ptr + get_smem_index<SMEM_M, SMEM_K, smem_A_skew, OP_A>{}(bm * FRAG_M, 0), get_smem_ld<SMEM_M, SMEM_K, smem_A_skew, OP_A>::value);
		cumpsgemm::device::load_matrix(frag_b[0], b_smem_ptr + get_smem_index<SMEM_K, SMEM_N, smem_B_skew, OP_B>{}(0, bn * FRAG_N), get_smem_ld<SMEM_K, SMEM_N, smem_B_skew, OP_B>::value);
		unsigned k = FRAG_K;
		auto frag_index = 1u;
		for (; k < SMEM_K; k += FRAG_K) {
			cumpsgemm::device::load_matrix(frag_a[frag_index], a_smem_ptr + get_smem_index<SMEM_M, SMEM_K, smem_A_skew, OP_A>{}(bm * FRAG_M, k), get_smem_ld<SMEM_M, SMEM_K, smem_A_skew, OP_A>::value);
			cumpsgemm::device::load_matrix(frag_b[frag_index], b_smem_ptr + get_smem_index<SMEM_K, SMEM_N, smem_B_skew, OP_B>{}(k, bn * FRAG_N), get_smem_ld<SMEM_K, SMEM_N, smem_B_skew, OP_B>::value);

			frag_index = 1 - frag_index;
			cumpsgemm::device::mma(frag_c[i / (BLOCK_SIZE / warp_size)],
					frag_a[frag_index],
					frag_b[frag_index],
					frag_c[i / (BLOCK_SIZE / warp_size)]);
		}
		frag_index = 1 - frag_index;
		cumpsgemm::device::mma(frag_c[i / (BLOCK_SIZE / warp_size)],
				frag_a[frag_index],
				frag_b[frag_index],
				frag_c[i / (BLOCK_SIZE / warp_size)]);
	}
}
};

template <
	class T,
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned BLOCK_SIZE,
	unsigned NUM_UNROLLINGS,
	unsigned NUM_STAGES,
	class A_DMEM_LOADER,
	class B_DMEM_LOADER,
	class C_DMEM_STORER,
	class MMA_SMEM,
	class TC_T,
	class EC
>
struct gemm_core;

template <
	class T,
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned BLOCK_SIZE,
	unsigned NUM_UNROLLINGS,
	class A_DMEM_LOADER,
	class B_DMEM_LOADER,
	class C_DMEM_STORER,
	class MMA_SMEM,
	class TC_T,
	class EC
>
struct gemm_core<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, NUM_UNROLLINGS, 2, A_DMEM_LOADER, B_DMEM_LOADER, C_DMEM_STORER, MMA_SMEM, TC_T, EC> {
__device__ void operator() (
		const unsigned m,
		const unsigned n,
		const unsigned k,
		const T alpha,
		const T* const a_dmem_ptr, const unsigned lda,
		const T* const b_dmem_ptr, const unsigned ldb,
		const T beta,
		T* const c_dmem_ptr, const unsigned ldc,
		const unsigned blockIdx_x, const unsigned blockIdx_y,
		const typename cumpsgemm::device::element_t_conv<T>::type ignore_threshold,
		const typename cumpsgemm::device::element_t_conv<T>::type lost_threshold,
		cumpsgemm::counter_t* const total_counter,
		cumpsgemm::counter_t* const lost_counter
		) {
	extern __shared__ uint8_t smem_base[];
	T* smem = reinterpret_cast<T*>(smem_base);
	T* const a_smem_ptr = smem;
	T* const b_smem_ptr = smem + get_smem_size<SMEM_M, SMEM_K, smem_A_skew, typename A_DMEM_LOADER::Layout>::value * 2;

	A_DMEM_LOADER a_dmem_loader;
	B_DMEM_LOADER b_dmem_loader;

	a_dmem_loader(
			a_smem_ptr,
			a_dmem_ptr,
			lda,
			blockIdx_x * SMEM_M, 0,
			m, k
			);
	cutf::cp_async::commit();
	b_dmem_loader(
			b_smem_ptr,
			b_dmem_ptr,
			ldb,
			0, blockIdx_y * SMEM_N,
			k, n
			);
	cutf::cp_async::commit();

	constexpr unsigned frag_c_array_size = SMEM_M * SMEM_N / (FRAG_M * FRAG_N) / (BLOCK_SIZE / warp_size);
	cumpsgemm::device::tc_fragment<T, nvcuda::wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, void, TC_T, EC> frag_c[frag_c_array_size];
	for (unsigned i = 0; i < frag_c_array_size; i++) {
		cumpsgemm::device::fill_zero(frag_c[i]);
	}

	cutf::cp_async::wait_all();
	__syncthreads();
	unsigned bk = SMEM_K;
#pragma unroll NUM_UNROLLINGS
	for (; bk < k; bk += SMEM_K) {
		const auto smem_buffer_id = (bk / SMEM_K) % 2;
		a_dmem_loader(
				a_smem_ptr + get_smem_size<SMEM_M, SMEM_K, smem_A_skew, typename A_DMEM_LOADER::Layout>::value * smem_buffer_id,
				a_dmem_ptr,
				lda,
				blockIdx_x * SMEM_M, bk,
				m, k
				);
		cutf::cp_async::commit();
		b_dmem_loader(
				b_smem_ptr + get_smem_size<SMEM_K, SMEM_N, smem_B_skew, typename B_DMEM_LOADER::Layout>::value * smem_buffer_id,
				b_dmem_ptr,
				ldb,
				bk, blockIdx_y * SMEM_N,
				k, n
				);
		cutf::cp_async::commit();
		MMA_SMEM{}(
					frag_c,
					a_smem_ptr + get_smem_size<SMEM_M, SMEM_K, smem_A_skew, typename A_DMEM_LOADER::Layout>::value * (1 - smem_buffer_id),
					b_smem_ptr + get_smem_size<SMEM_K, SMEM_N, smem_B_skew, typename B_DMEM_LOADER::Layout>::value * (1 - smem_buffer_id)
				 );
		cutf::cp_async::wait_all();
		__syncthreads();
	}
	const auto smem_buffer_id = 1 - ((bk / SMEM_K) % 2);
	MMA_SMEM{}(
				frag_c,
				a_smem_ptr + get_smem_size<SMEM_M, SMEM_K, smem_A_skew, typename A_DMEM_LOADER::Layout>::value * smem_buffer_id,
				b_smem_ptr + get_smem_size<SMEM_K, SMEM_N, smem_B_skew, typename B_DMEM_LOADER::Layout>::value * smem_buffer_id
			 );
	__syncthreads();

	// register to smem
	for (unsigned i = threadIdx.x / warp_size; i < (SMEM_M / FRAG_M) * (SMEM_N / FRAG_N); i += BLOCK_SIZE / warp_size) {
		const unsigned bm = i % (SMEM_M / FRAG_M);
		const unsigned bn = i / (SMEM_M / FRAG_M);
		cumpsgemm::device::store_matrix(
				smem + get_smem_index<SMEM_M, SMEM_N, smem_C_skew, cumpsgemm::col_major>{}(
					bm * FRAG_M, bn * FRAG_N
					),
				frag_c[i / (BLOCK_SIZE / warp_size)],
				SMEM_M + smem_C_skew
				);
	}
	__syncthreads();

	C_DMEM_STORER c_dmem_storer;
	if (total_counter == nullptr) {
		c_dmem_storer(
				c_dmem_ptr, ldc,
				blockIdx_x * SMEM_M, blockIdx_y * SMEM_N,
				m, n,
				smem,
				alpha, beta,
				0, 0,
				nullptr, nullptr
				);
		return;
	}

	unsigned local_total_counter = 0;
	unsigned local_lost_counter  = 0;

	c_dmem_storer(
			c_dmem_ptr, ldc,
			blockIdx_x * SMEM_M, blockIdx_y * SMEM_N,
			m, n,
			smem,
			alpha, beta,
			ignore_threshold, lost_threshold,
			&local_total_counter, &local_lost_counter
			);

	for (std::uint32_t offset = warp_size >> 1; offset >= 1; offset >>= 1) {
		local_lost_counter  += __shfl_xor_sync(~0u, local_lost_counter , offset);
		local_total_counter += __shfl_xor_sync(~0u, local_total_counter, offset);
	}

	unsigned *smem_lost_counter_ptr = reinterpret_cast<unsigned*>(smem);
	unsigned *smem_total_counter_ptr  = smem_lost_counter_ptr + (BLOCK_SIZE / warp_size);

	if ((threadIdx.x & 0x1f) == 0) {
		smem_lost_counter_ptr [threadIdx.x >> 5] = local_lost_counter;
		smem_total_counter_ptr[threadIdx.x >> 5] = local_total_counter;
	}
	__syncthreads();

	if (threadIdx.x >= BLOCK_SIZE / warp_size) return;

	local_total_counter = smem_lost_counter_ptr [threadIdx.x];
	local_lost_counter  = smem_total_counter_ptr[threadIdx.x];

	for (std::uint32_t offset = (BLOCK_SIZE / warp_size) >> 1; offset >= 1; offset >>= 1) {
		local_lost_counter  += __shfl_xor_sync(~0u, local_lost_counter , offset);
		local_total_counter += __shfl_xor_sync(~0u, local_total_counter, offset);
	}

	if (threadIdx.x == 0) {
		atomicAdd(lost_counter , local_lost_counter);
		atomicAdd(total_counter, local_total_counter);
	}
}
};

template <
	class T,
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned BLOCK_SIZE,
	unsigned NUM_UNROLLINGS,
	unsigned NUM_STAGES,
	class A_DMEM_LOADER,
	class B_DMEM_LOADER,
	class C_DMEM_STORER,
	class MMA_SMEM,
	class TC_T,
	class EC
>
__global__ void gemm_kernel(
		const unsigned m,
		const unsigned n,
		const unsigned k,
		const T alpha,
		const T* const a_dmem_ptr, const unsigned lda,
		const T* const b_dmem_ptr, const unsigned ldb,
		const T beta,
		T* const c_dmem_ptr, const unsigned ldc,
		const typename cumpsgemm::device::element_t_conv<T>::type ignore_threshold,
		const typename cumpsgemm::device::element_t_conv<T>::type lost_threshold,
		cumpsgemm::counter_t* const total_counter,
		cumpsgemm::counter_t* const lost_counter
		) {
	const auto blockIdx_x = (blockIdx.x) % ((m + SMEM_M - 1) / SMEM_M);
	const auto blockIdx_y = (blockIdx.x) / ((m + SMEM_M - 1) / SMEM_M);

	gemm_core<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, NUM_UNROLLINGS, NUM_STAGES, A_DMEM_LOADER, B_DMEM_LOADER, C_DMEM_STORER, MMA_SMEM, TC_T, EC>{}(
			m, n, k,
			alpha,
			a_dmem_ptr, lda,
			b_dmem_ptr, ldb,
			beta,
			c_dmem_ptr, ldc,
			blockIdx_x, blockIdx_y,
			ignore_threshold, lost_threshold,
			total_counter, lost_counter
			);
}

template <
	class T,
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned BLOCK_SIZE,
	unsigned NUM_UNROLLINGS,
	unsigned NUM_STAGES,
	class A_DMEM_LOADER,
	class B_DMEM_LOADER,
	class C_DMEM_STORER,
	class MMA_SMEM,
	class TC_T,
	class EC
>
__global__ void gemm_batchStrided_kernel(
		const unsigned m,
		const unsigned n,
		const unsigned k,
		const T alpha,
		const T* const a_ptr, const unsigned lda, const uint64_t stridea,
		const T* const b_ptr, const unsigned ldb, const uint64_t strideb,
		const T beta,
		T* const c_ptr, const unsigned ldc, const uint64_t stridec,
		const unsigned num_blocks_per_gemm,
		const typename cumpsgemm::device::element_t_conv<T>::type ignore_threshold,
		const typename cumpsgemm::device::element_t_conv<T>::type lost_threshold,
		cumpsgemm::counter_t* const total_counter,
		cumpsgemm::counter_t* const lost_counter
		) {
	const auto gemm_id = blockIdx.x / num_blocks_per_gemm;
	const auto blockIdx_x = (blockIdx.x % num_blocks_per_gemm) % ((m + SMEM_M - 1) / SMEM_M);
	const auto blockIdx_y = (blockIdx.x % num_blocks_per_gemm) / ((m + SMEM_M - 1) / SMEM_M);

	const T* const a_dmem_ptr = a_ptr + gemm_id * stridea;
	const T* const b_dmem_ptr = b_ptr + gemm_id * strideb;
	T* const c_dmem_ptr = c_ptr + gemm_id * stridec;

	gemm_core<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, NUM_UNROLLINGS, NUM_STAGES, A_DMEM_LOADER, B_DMEM_LOADER, C_DMEM_STORER, MMA_SMEM, TC_T, EC>{}(
			m, n, k,
			alpha,
			a_dmem_ptr, lda,
			b_dmem_ptr, ldb,
			beta,
			c_dmem_ptr, ldc,
			blockIdx_x, blockIdx_y,
			ignore_threshold, lost_threshold,
			total_counter + gemm_id, lost_counter + gemm_id
			);
}


template <
	class T,
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	class OP_A,
	class OP_B,
	unsigned NUM_STAGES
	>
unsigned get_total_smem_size() {
	return sizeof(T) * std::max<unsigned>(
			(SMEM_M + smem_C_skew) * SMEM_N,
			NUM_STAGES * (
				get_smem_size<SMEM_M, SMEM_K, smem_A_skew, OP_A>::value +
				get_smem_size<SMEM_K, SMEM_N, smem_B_skew, OP_B>::value
				));
}

template <
	class T,
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned BLOCK_SIZE,
	unsigned NUM_UNROLLINGS,
	unsigned NUM_STAGES,
	class OP_A,
	class OP_B,
	class TC_T,
	class EC
>
cumpsgemm::gemm_kernel_func_t<T> get_kernel_func_ptr() {
	using A_DMEM_LOADER = cumpsgemm::device::dmem_loader<OP_A, T, SMEM_M, SMEM_K, smem_A_skew, BLOCK_SIZE>;
	using B_DMEM_LOADER = cumpsgemm::device::dmem_loader<OP_B, T, SMEM_K, SMEM_N, smem_B_skew, BLOCK_SIZE>;
	using C_DMEM_STORER = cumpsgemm::device::dmem_storer<T, SMEM_M, SMEM_N, smem_C_skew, BLOCK_SIZE>;
	constexpr cumpsgemm::gemm_kernel_func_t<T> func_ptr = &(gemm_kernel<
		T,
		SMEM_M, SMEM_N, SMEM_K,
		FRAG_M, FRAG_N, FRAG_K,
		BLOCK_SIZE,
		NUM_UNROLLINGS,
		NUM_STAGES,
		A_DMEM_LOADER,
		B_DMEM_LOADER,
		C_DMEM_STORER,
		mma_smem<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, typename A_DMEM_LOADER::Layout, typename B_DMEM_LOADER::Layout, TC_T, EC>,
		TC_T,
		EC
	>);
	return func_ptr;
}

template <
	class T,
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned BLOCK_SIZE,
	unsigned NUM_UNROLLINGS,
	unsigned NUM_STAGES,
	class OP_A,
	class OP_B,
	class TC_T,
	class EC
>
cumpsgemm::gemm_kernel_func_t<T> get_kernel_pipelined_func_ptr() {
	using A_DMEM_LOADER = cumpsgemm::device::dmem_loader<OP_A, T, SMEM_M, SMEM_K, smem_A_skew, BLOCK_SIZE>;
	using B_DMEM_LOADER = cumpsgemm::device::dmem_loader<OP_B, T, SMEM_K, SMEM_N, smem_B_skew, BLOCK_SIZE>;
	using C_DMEM_STORER = cumpsgemm::device::dmem_storer<T, SMEM_M, SMEM_N, smem_C_skew, BLOCK_SIZE>;
	constexpr cumpsgemm::gemm_kernel_func_t<T> func_ptr = &(gemm_kernel<
		T,
		SMEM_M, SMEM_N, SMEM_K,
		FRAG_M, FRAG_N, FRAG_K,
		BLOCK_SIZE,
		NUM_UNROLLINGS,
		NUM_STAGES,
		cumpsgemm::device::dmem_loader<OP_A, T, SMEM_M, SMEM_K, smem_A_skew, BLOCK_SIZE>,
		cumpsgemm::device::dmem_loader<OP_B, T, SMEM_K, SMEM_N, smem_B_skew, BLOCK_SIZE>,
		cumpsgemm::device::dmem_storer<T, SMEM_M, SMEM_N, smem_C_skew, BLOCK_SIZE>,
		mma_smem_pipeline<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, typename A_DMEM_LOADER::Layout, typename B_DMEM_LOADER::Layout, TC_T, EC>,
		TC_T,
		EC
	>);
	return func_ptr;
}

template <
	class T,
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned BLOCK_SIZE,
	unsigned NUM_UNROLLINGS,
	unsigned NUM_STAGES,
	class OP_A,
	class OP_B,
	class TC_T,
	class EC
>
cumpsgemm::gemm_stridedBatch_kernel_func_t<T> get_stridedBatch_kernel_func_ptr() {
	using A_DMEM_LOADER = cumpsgemm::device::dmem_loader<OP_A, T, SMEM_M, SMEM_K, smem_A_skew, BLOCK_SIZE>;
	using B_DMEM_LOADER = cumpsgemm::device::dmem_loader<OP_B, T, SMEM_K, SMEM_N, smem_B_skew, BLOCK_SIZE>;
	using C_DMEM_STORER = cumpsgemm::device::dmem_storer<T, SMEM_M, SMEM_N, smem_C_skew, BLOCK_SIZE>;
	constexpr cumpsgemm::gemm_stridedBatch_kernel_func_t<T> func_ptr = &(gemm_batchStrided_kernel<
		T,
		SMEM_M, SMEM_N, SMEM_K,
		FRAG_M, FRAG_N, FRAG_K,
		BLOCK_SIZE,
		NUM_UNROLLINGS,
		NUM_STAGES,
		cumpsgemm::device::dmem_loader<OP_A, T, SMEM_M, SMEM_K, smem_A_skew, BLOCK_SIZE>,
		cumpsgemm::device::dmem_loader<OP_B, T, SMEM_K, SMEM_N, smem_B_skew, BLOCK_SIZE>,
		cumpsgemm::device::dmem_storer<T, SMEM_M, SMEM_N, smem_C_skew, BLOCK_SIZE>,
		mma_smem<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, typename A_DMEM_LOADER::Layout, typename B_DMEM_LOADER::Layout, TC_T, EC>,
		TC_T,
		EC
	>);
	return func_ptr;
}

template <
	class T,
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned BLOCK_SIZE,
	unsigned NUM_UNROLLINGS,
	unsigned NUM_STAGES,
	class OP_A,
	class OP_B,
	class TC_T,
	class EC
>
cumpsgemm::gemm_stridedBatch_kernel_func_t<T> get_stridedBatch_kernel_pipelined_func_ptr() {
	using A_DMEM_LOADER = cumpsgemm::device::dmem_loader<OP_A, T, SMEM_M, SMEM_K, smem_A_skew, BLOCK_SIZE>;
	using B_DMEM_LOADER = cumpsgemm::device::dmem_loader<OP_B, T, SMEM_K, SMEM_N, smem_B_skew, BLOCK_SIZE>;
	using C_DMEM_STORER = cumpsgemm::device::dmem_storer<T, SMEM_M, SMEM_N, smem_C_skew, BLOCK_SIZE>;
	constexpr cumpsgemm::gemm_stridedBatch_kernel_func_t<T> func_ptr = &(gemm_batchStrided_kernel<
		T,
		SMEM_M, SMEM_N, SMEM_K,
		FRAG_M, FRAG_N, FRAG_K,
		BLOCK_SIZE,
		NUM_UNROLLINGS,
		NUM_STAGES,
		cumpsgemm::device::dmem_loader<OP_A, T, SMEM_M, SMEM_K, smem_A_skew, BLOCK_SIZE>,
		cumpsgemm::device::dmem_loader<OP_B, T, SMEM_K, SMEM_N, smem_B_skew, BLOCK_SIZE>,
		cumpsgemm::device::dmem_storer<T, SMEM_M, SMEM_N, smem_C_skew, BLOCK_SIZE>,
		mma_smem_pipeline<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, typename A_DMEM_LOADER::Layout, typename B_DMEM_LOADER::Layout, TC_T, EC>,
		TC_T,
		EC
	>);
	return func_ptr;
}
} // noname namespace

namespace cumpsgemm {
template <
	class T,
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned BLOCK_SIZE,
	unsigned NUM_UNROLLINGS,
	unsigned NUM_STAGES,
	class OP_A,
	class OP_B,
	class TC_T,
	class EC,
	bool PIPELINED
>
cumpsgemm::gemm_module generate_gemm_module() {
	cumpsgemm::gemm_kernel_func_t<T> kernel_func;
	if constexpr (PIPELINED) {
		kernel_func = get_kernel_pipelined_func_ptr<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, NUM_UNROLLINGS, NUM_STAGES, OP_A, OP_B, TC_T, EC>();
	} else {
		kernel_func = get_kernel_func_ptr<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, NUM_UNROLLINGS, NUM_STAGES, OP_A, OP_B, TC_T, EC>();
	}
	cumpsgemm::gemm_module mod;
	mod.kernel_func = reinterpret_cast<void*>(kernel_func);
	mod.block_size = BLOCK_SIZE;
	mod.smem_size = get_total_smem_size<T, SMEM_M, SMEM_N, SMEM_K, OP_A, OP_B, NUM_STAGES>();
	mod.smem_m = SMEM_M;
	mod.smem_n = SMEM_N;
	mod.smem_k = SMEM_K;
	CUTF_CHECK_ERROR_M(cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, mod.smem_size), ("requested shared memory size = " + std::to_string(mod.smem_size) + " [B]").c_str());

	int num_active_blocks;
	CUTF_CHECK_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_active_blocks, kernel_func, BLOCK_SIZE, mod.smem_size));
	mod.num_active_blocks = num_active_blocks;

	return mod;
}

template <
	class T,
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	unsigned FRAG_M,
	unsigned FRAG_N,
	unsigned FRAG_K,
	unsigned BLOCK_SIZE,
	unsigned NUM_UNROLLINGS,
	unsigned NUM_STAGES,
	class OP_A,
	class OP_B,
	class TC_T,
	class EC,
	bool PIPELINED
>
cumpsgemm::gemm_module generate_gemm_stridedBatch_module() {
	cumpsgemm::gemm_stridedBatch_kernel_func_t<T> kernel_func;
	if constexpr (PIPELINED) {
		kernel_func = get_stridedBatch_kernel_pipelined_func_ptr<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, NUM_UNROLLINGS, NUM_STAGES, OP_A, OP_B, TC_T, EC>();
	} else {
		kernel_func = get_stridedBatch_kernel_func_ptr<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, NUM_UNROLLINGS, NUM_STAGES, OP_A, OP_B, TC_T, EC>();
	}
	cumpsgemm::gemm_module mod;
	mod.kernel_func = reinterpret_cast<void*>(kernel_func);
	mod.block_size = BLOCK_SIZE;
	mod.smem_size = get_total_smem_size<T, SMEM_M, SMEM_N, SMEM_K, OP_A, OP_B, NUM_STAGES>();
	mod.smem_m = SMEM_M;
	mod.smem_n = SMEM_N;
	mod.smem_k = SMEM_K;
	CUTF_CHECK_ERROR_M(cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, mod.smem_size), ("requested shared memory size = " + std::to_string(mod.smem_size) + " [B]").c_str());

	int num_active_blocks;
	CUTF_CHECK_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_active_blocks, kernel_func, BLOCK_SIZE, mod.smem_size));
	mod.num_active_blocks = num_active_blocks;

	return mod;
}
} // noname namespace
#endif
