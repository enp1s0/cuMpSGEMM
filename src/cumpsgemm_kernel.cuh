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
#include "cumpsgemm_internal.hpp"
#include "handle.hpp"

namespace {
constexpr unsigned smem_A_skew = 8;
constexpr unsigned smem_B_skew = 8;
constexpr unsigned smem_C_skew = 8;
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

// zero
template <class T>
__device__ T zero() {return 0;}
template <> __device__ cuComplex zero<cuComplex>() {return make_cuComplex(0, 0);}

template <class T>
struct size_of {static constexpr unsigned value = 0;};
template <> struct size_of<float    > {static constexpr unsigned value = 4;};
template <> struct size_of<cuComplex> {static constexpr unsigned value = 8;};

// Dmem loader
template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, unsigned BLOCK_SIZE>
struct dmem_loader_core {
	__device__ dmem_loader_core(){}
	__device__ void operator() (
			T* const smem_ptr,
			const T* const dmem_ptr,
			const std::size_t ld,
			const std::size_t start_m,
			const std::size_t start_n,
			const std::size_t size_m,
			const std::size_t size_n
			) {
		if (start_m + SMEM_M < size_m && start_n + SMEM_N < size_n) {
			if (ld % (16 / size_of<T>::value) == 0) {
				for (unsigned offset = 0; offset < SMEM_M * SMEM_N; offset += BLOCK_SIZE * (16 / size_of<T>::value)) {
					const auto index = offset + threadIdx.x * (16 / size_of<T>::value);
					const auto m = index % SMEM_M;
					const auto n = index / SMEM_M;
					const auto smem_index = m + n * (SMEM_M + SKEW);
					const auto dmem_index = (start_m + m) + (start_n + n) * ld;
					cutf::cp_async::cp_async<16>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				}
			} else if ((ld % (8 / size_of<T>::value) == 0)) {
				for (unsigned offset = 0; offset < SMEM_M * SMEM_N; offset += BLOCK_SIZE * (8 / size_of<T>::value)) {
					const auto index = offset + threadIdx.x * (8 / size_of<T>::value);
					const auto m = index % SMEM_M;
					const auto n = index / SMEM_M;
					const auto smem_index = m + n * (SMEM_M + SKEW);
					const auto dmem_index = (start_m + m) + (start_n + n) * ld;
					cutf::cp_async::cp_async<8>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				}
			} else if ((4 / size_of<T>::value != 0) && (ld % (4 / size_of<T>::value) == 0)) {
				for (unsigned offset = 0; offset < SMEM_M * SMEM_N; offset += BLOCK_SIZE * (4 / size_of<T>::value)) {
					const auto index = offset + threadIdx.x * (4 / size_of<T>::value);
					const auto m = index % SMEM_M;
					const auto n = index / SMEM_M;
					const auto smem_index = m + n * (SMEM_M + SKEW);
					const auto dmem_index = (start_m + m) + (start_n + n) * ld;
					cutf::cp_async::cp_async<4>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				}
			}
		} else {
			for (unsigned offset = 0; offset < SMEM_M * SMEM_N; offset += BLOCK_SIZE) {
				const auto index = offset + threadIdx.x;
				const auto m = index % SMEM_M;
				const auto n = index / SMEM_M;
				const auto smem_index = m + n * (SMEM_M + SKEW);
				const auto dmem_index = (start_m + m) + (start_n + n) * ld;

				T v = zero<T>();
				if ((start_m + m) < size_m && (start_n + n) < size_n) {
					v = dmem_ptr[dmem_index];
				}
				__syncwarp();
				smem_ptr[smem_index] = v;
			}
		}
		cutf::cp_async::commit();
	}
};

template <class _Layout, class T, unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, unsigned BLOCK_SIZE>
struct dmem_loader {
	using Layout = _Layout;
	__device__ dmem_loader(){}
	__device__ void operator() (
			T* const smem_ptr,
			const T* const dmem_ptr,
			const std::size_t ld,
			const std::size_t start_m,
			const std::size_t start_n,
			const std::size_t size_m,
			const std::size_t size_n
			) {
		dmem_loader_core<T, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE>{}(
				smem_ptr,
				dmem_ptr,
				ld,
				start_m, start_n,
				size_m, size_n
				);
	}
};

template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, unsigned BLOCK_SIZE>
struct dmem_loader<cumpsgemm::row_major, T, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE> {
	using Layout = cumpsgemm::row_major;
	__device__ dmem_loader(){}
	__device__ void operator() (
			T* const smem_ptr,
			const T* const dmem_ptr,
			const std::size_t ld,
			const std::size_t start_m,
			const std::size_t start_n,
			const std::size_t size_m,
			const std::size_t size_n
			) {
		dmem_loader_core<T, SMEM_N, SMEM_M, SKEW, BLOCK_SIZE>{}(
				smem_ptr,
				dmem_ptr,
				ld,
				start_n, start_m,
				size_n, size_m
				);
	}
};

template <unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, unsigned BLOCK_SIZE>
struct dmem_loader_conj_core {
	__device__ dmem_loader_conj_core(){}
	__device__ void operator() (
			cuComplex* const smem_ptr,
			const cuComplex* const dmem_ptr,
			const std::size_t ld,
			const std::size_t start_m,
			const std::size_t start_n,
			const std::size_t size_m,
			const std::size_t size_n
			) {
		if (start_m + SMEM_M < size_m && start_n + SMEM_N < size_n) {
			for (unsigned offset = 0; offset < SMEM_M * SMEM_N; offset += BLOCK_SIZE) {
				const auto index = offset + threadIdx.x;
				const auto m = index % SMEM_M;
				const auto n = index / SMEM_M;
				const auto smem_index = m + n * (SMEM_M + SKEW);
				const auto dmem_index = (start_m + m) + (start_n + n) * ld;
				const auto v = dmem_ptr[dmem_index];
				smem_ptr[smem_index] = make_cuComplex(v.x, -v.y);
			}
		} else {
			for (unsigned offset = 0; offset < SMEM_M * SMEM_N; offset += BLOCK_SIZE) {
				const auto index = offset + threadIdx.x;
				const auto m = index % SMEM_M;
				const auto n = index / SMEM_M;
				const auto smem_index = m + n * (SMEM_M + SKEW);
				const auto dmem_index = (start_m + m) + (start_n + n) * ld;

				auto v = zero<cuComplex>();
				if ((start_m + m) < size_m && (start_n + n) < size_n) {
					const auto w = dmem_ptr[dmem_index];
					v = make_cuComplex(w.x, -w.y);
				}
				__syncwarp();
				smem_ptr[smem_index] = v;
			}
		}
	}
};

template <unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, unsigned BLOCK_SIZE>
struct dmem_loader<cumpsgemm::conjugate, cuComplex, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE> {
	using Layout = cumpsgemm::row_major;
	__device__ dmem_loader(){}
	__device__ void operator() (
			cuComplex* const smem_ptr,
			const cuComplex* const dmem_ptr,
			const std::size_t ld,
			const std::size_t start_m,
			const std::size_t start_n,
			const std::size_t size_m,
			const std::size_t size_n
			) {
		dmem_loader_conj_core<SMEM_N, SMEM_M, SKEW, BLOCK_SIZE>{}(
				smem_ptr,
				dmem_ptr,
				ld,
				start_n, start_m,
				size_n, size_m
				);
	}
};

template <unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, unsigned BLOCK_SIZE>
struct dmem_loader<cumpsgemm::conjugate, float, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE> {
	using Layout = cumpsgemm::col_major;
	__device__ dmem_loader(){}
	__device__ void operator() (
			float* const,
			const float* const,
			const std::size_t,
			const std::size_t,
			const std::size_t,
			const std::size_t,
			const std::size_t
			) {
		// Do nothing, only for suppressing compilation error.
	}
};

template <class T>
__device__ T mul(const T a, const T alpha) {
	return a * alpha;
}
template <>
__device__ cuComplex mul<cuComplex>(const cuComplex a, const cuComplex alpha) {
	return make_cuComplex(a.x * alpha.x - a.y * alpha.y, a.y * alpha.x + a.x * alpha.y);
}

template <class T>
__device__ T mad(const T a, const T alpha, const T b) {
	return a * alpha + b;
}
template <>
__device__ cuComplex mad<cuComplex>(const cuComplex a, const cuComplex alpha, const cuComplex b) {
	return make_cuComplex(
			a.x * alpha.x - a.y * alpha.y + b.x,
			a.y * alpha.x + a.x * alpha.y + b.y
			);
}

template<class T>
__device__ bool is_zero(const T& v) {
	return v == 0;
}
template <>
__device__ bool is_zero(const cuComplex& v) {
	return v.x == 0 && v.y == 0;
}

template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, unsigned BLOCK_SIZE>
struct dmem_storer {
	__device__ dmem_storer(){}
	__device__ void operator() (
			T* const dmem_ptr,
			const std::size_t ld,
			const std::size_t start_m,
			const std::size_t start_n,
			const std::size_t size_m,
			const std::size_t size_n,
			const T* const smem_ptr,
			const T alpha, const T beta
			) {
		if (is_zero(beta)) {
			if (start_m + SMEM_M < size_m && start_n + SMEM_N < size_n) {
				for (unsigned offset = 0; offset < SMEM_M * SMEM_N; offset += BLOCK_SIZE) {
					const auto index = offset + threadIdx.x;
					const auto m = index % SMEM_M;
					const auto n = index / SMEM_M;
					const auto smem_index = m + n * (SMEM_M + SKEW);
					const auto dmem_index = (start_m + m) + (start_n + n) * ld;
					dmem_ptr[dmem_index] = mul(smem_ptr[smem_index], alpha);
				}
			} else {
				for (unsigned offset = 0; offset < SMEM_M * SMEM_N; offset += BLOCK_SIZE) {
					const auto index = offset + threadIdx.x;
					const auto m = index % SMEM_M;
					const auto n = index / SMEM_M;
					const auto smem_index = m + n * (SMEM_M + SKEW);
					const auto dmem_index = (start_m + m) + (start_n + n) * ld;

					if ((start_m + m) < size_m && (start_n + n) < size_n) {
						dmem_ptr[dmem_index] = mul(smem_ptr[smem_index], alpha);
					}
					__syncwarp();
				}
			}
		} else {
			if (start_m + SMEM_M < size_m && start_n + SMEM_N < size_n) {
				for (unsigned offset = 0; offset < SMEM_M * SMEM_N; offset += BLOCK_SIZE) {
					const auto index = offset + threadIdx.x;
					const auto m = index % SMEM_M;
					const auto n = index / SMEM_M;
					const auto smem_index = m + n * (SMEM_M + SKEW);
					const auto dmem_index = (start_m + m) + (start_n + n) * ld;
					dmem_ptr[dmem_index] = mad(smem_ptr[smem_index], alpha, mul(dmem_ptr[dmem_index], beta));
				}
			} else {
				for (unsigned offset = 0; offset < SMEM_M * SMEM_N; offset += BLOCK_SIZE) {
					const auto index = offset + threadIdx.x;
					const auto m = index % SMEM_M;
					const auto n = index / SMEM_M;
					const auto smem_index = m + n * (SMEM_M + SKEW);
					const auto dmem_index = (start_m + m) + (start_n + n) * ld;

					if ((start_m + m) < size_m && (start_n + n) < size_n) {
						dmem_ptr[dmem_index] = mad(smem_ptr[smem_index], alpha, mul(dmem_ptr[dmem_index], beta));
					}
					__syncwarp();
				}
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
__device__ void mma_smem (
		cumpsgemm::device::tc_fragment<T, nvcuda::wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, void, TC_T, EC> frag_c[SMEM_M * SMEM_N / (FRAG_M * FRAG_N) / (BLOCK_SIZE / warp_size)],
		const T* const a_smem_ptr,
		const T* const b_smem_ptr
		) {
	static_assert((SMEM_M / FRAG_M) * (SMEM_N / FRAG_N) >= (BLOCK_SIZE / warp_size));
	for (unsigned i = threadIdx.x / warp_size; i < (SMEM_M / FRAG_M) * (SMEM_N / FRAG_N); i += BLOCK_SIZE / warp_size) {
		const unsigned bm = i % (SMEM_M / FRAG_M);
		const unsigned bn = i / (SMEM_M / FRAG_M);

		for (unsigned k = 0; k < SMEM_K; k += FRAG_K) {
			cumpsgemm::device::tc_fragment<T, nvcuda::wmma::matrix_a, FRAG_M, FRAG_N, FRAG_K, OP_A, TC_T, EC> frag_a;
			cumpsgemm::device::load_matrix<OP_A>(frag_a, a_smem_ptr + get_smem_index<SMEM_M, SMEM_K, smem_A_skew, OP_A>{}(bm * FRAG_M, k), get_smem_ld<SMEM_M, SMEM_K, smem_A_skew, OP_A>::value);

			cumpsgemm::device::tc_fragment<T, nvcuda::wmma::matrix_b, FRAG_M, FRAG_N, FRAG_K, OP_B, TC_T, EC> frag_b;
			cumpsgemm::device::load_matrix<OP_B>(frag_b, b_smem_ptr + get_smem_index<SMEM_K, SMEM_N, smem_B_skew, OP_B>{}(k, bn * FRAG_N), get_smem_ld<SMEM_K, SMEM_N, smem_B_skew, OP_B>::value);

			cumpsgemm::device::mma(frag_c[i], frag_a, frag_b, frag_c[i]);
		}
	}
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
	class A_DMEM_LOADER,
	class B_DMEM_LOADER,
	class C_DMEM_STORER,
	class TC_T,
	class EC
>
__global__ void gemm_kernel(
		const uint32_t m,
		const uint32_t n,
		const uint32_t k,
		const T alpha,
		const T* const a_dmem_ptr, const uint32_t lda,
		const T* const b_dmem_ptr, const uint32_t ldb,
		const T beta,
		T* const c_dmem_ptr, const uint32_t ldc
		) {
	extern __shared__ uint8_t smem_base[];
	T* smem = reinterpret_cast<T*>(smem_base);
	T* const a_smem_ptr = smem;
	T* const b_smem_ptr = smem + get_smem_size<SMEM_M, SMEM_K, smem_A_skew, typename A_DMEM_LOADER::Layout>::value * 2;

	A_DMEM_LOADER a_dmem_loader;
	B_DMEM_LOADER b_dmem_loader;

	constexpr unsigned frag_c_array_size = SMEM_M * SMEM_N / (FRAG_M * FRAG_N) / (BLOCK_SIZE / warp_size);
	cumpsgemm::device::tc_fragment<T, nvcuda::wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, void, TC_T, EC> frag_c[frag_c_array_size];
	for (unsigned i = 0; i < frag_c_array_size; i++) {
		cumpsgemm::device::fill_zero(frag_c[i]);
	}

	const auto blockIdx_x = (blockIdx.x) % ((m + SMEM_M - 1) / SMEM_M);
	const auto blockIdx_y = (blockIdx.x) / ((m + SMEM_M - 1) / SMEM_M);

	a_dmem_loader(
			a_smem_ptr,
			a_dmem_ptr,
			lda,
			blockIdx_x * SMEM_M, 0,
			m, k
			);
	b_dmem_loader(
			b_smem_ptr,
			b_dmem_ptr,
			ldb,
			0, blockIdx_y * SMEM_N,
			k, n
			);
	unsigned bk = 0;
#pragma unroll NUM_UNROLLINGS
	for (bk += SMEM_K; bk < k; bk += SMEM_K) {
		const auto smem_buffer_id = (bk / SMEM_K) % 2;
		a_dmem_loader(
				a_smem_ptr + get_smem_size<SMEM_M, SMEM_K, smem_A_skew, typename A_DMEM_LOADER::Layout>::value * smem_buffer_id,
				a_dmem_ptr,
				lda,
				blockIdx_x * SMEM_M, bk,
				m, k
				);
		b_dmem_loader(
				b_smem_ptr + get_smem_size<SMEM_K, SMEM_N, smem_B_skew, typename B_DMEM_LOADER::Layout>::value * smem_buffer_id,
				b_dmem_ptr,
				ldb,
				bk, blockIdx_y * SMEM_N,
				k, n
				);
		cutf::cp_async::wait_group<2>();
		__syncthreads();
		mma_smem<
			T,
			SMEM_M, SMEM_N, SMEM_K,
			FRAG_M, FRAG_N, FRAG_K,
			BLOCK_SIZE,
			typename A_DMEM_LOADER::Layout,
			typename B_DMEM_LOADER::Layout,
			TC_T,
			EC>(
					frag_c,
					a_smem_ptr + get_smem_size<SMEM_M, SMEM_K, smem_A_skew, typename A_DMEM_LOADER::Layout>::value * (1 - smem_buffer_id),
					b_smem_ptr + get_smem_size<SMEM_K, SMEM_N, smem_B_skew, typename B_DMEM_LOADER::Layout>::value * (1 - smem_buffer_id)
				 );
		__syncthreads();
	}
	const auto smem_buffer_id = 1 - ((bk / SMEM_K) % 2);
	cutf::cp_async::wait_all();
	__syncthreads();
	mma_smem<
		T,
		SMEM_M, SMEM_N, SMEM_K,
		FRAG_M, FRAG_N, FRAG_K,
		BLOCK_SIZE,
		typename A_DMEM_LOADER::Layout,
		typename B_DMEM_LOADER::Layout,
		TC_T,
		EC>(
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
				frag_c[i],
				SMEM_M + smem_C_skew
				);
	}
	__syncthreads();

	C_DMEM_STORER c_dmem_storer;
	c_dmem_storer(
			c_dmem_ptr, ldc,
			blockIdx_x * SMEM_M, blockIdx_y * SMEM_N,
			m, n,
			smem,
			alpha, beta
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
	class A_DMEM_LOADER,
	class B_DMEM_LOADER,
	class C_DMEM_STORER,
	class TC_T,
	class EC
>
__global__ void gemm_batchStrided_kernel(
		const uint32_t m,
		const uint32_t n,
		const uint32_t k,
		const T alpha,
		const T* const a_ptr, const uint32_t lda, const uint64_t stridea,
		const T* const b_ptr, const uint32_t ldb, const uint64_t strideb,
		const T beta,
		T* const c_ptr, const uint32_t ldc, const uint64_t stridec,
		const unsigned num_blocks_per_gemm
		) {
	extern __shared__ uint8_t smem_base[];
	T* smem = reinterpret_cast<T*>(smem_base);
	T* const a_smem_ptr = smem;
	T* const b_smem_ptr = smem + get_smem_size<SMEM_M, SMEM_K, smem_A_skew, typename A_DMEM_LOADER::Layout>::value * 2;

	A_DMEM_LOADER a_dmem_loader;
	B_DMEM_LOADER b_dmem_loader;

	const auto gemm_id = blockIdx.x / num_blocks_per_gemm;
	const auto blockIdx_x = (blockIdx.x % num_blocks_per_gemm) % ((m + SMEM_M - 1) / SMEM_M);
	const auto blockIdx_y = (blockIdx.x % num_blocks_per_gemm) / ((m + SMEM_M - 1) / SMEM_M);

	const T* const a_dmem_ptr = a_ptr + gemm_id * stridea;
	const T* const b_dmem_ptr = b_ptr + gemm_id * strideb;
	T* const c_dmem_ptr = c_ptr + gemm_id * stridec;

	constexpr unsigned frag_c_array_size = SMEM_M * SMEM_N / (FRAG_M * FRAG_N) / (BLOCK_SIZE / warp_size);
	cumpsgemm::device::tc_fragment<T, nvcuda::wmma::accumulator, FRAG_M, FRAG_N, FRAG_K, void, TC_T, EC> frag_c[frag_c_array_size];
	for (unsigned i = 0; i < frag_c_array_size; i++) {
		cumpsgemm::device::fill_zero(frag_c[i]);
	}

	std::uint32_t bk = 0;
	a_dmem_loader(
			a_smem_ptr,
			a_dmem_ptr,
			lda,
			blockIdx_x * SMEM_M, 0,
			m, k
			);
	b_dmem_loader(
			b_smem_ptr,
			b_dmem_ptr,
			ldb,
			0, blockIdx_y * SMEM_N,
			k, n
			);
#pragma NUM_UNROLLINGS
	for (bk += SMEM_K; bk < k; bk += SMEM_K) {
		const auto smem_buffer_id = (bk / SMEM_K) % 2;
		a_dmem_loader(
				a_smem_ptr + get_smem_size<SMEM_M, SMEM_K, smem_A_skew, typename A_DMEM_LOADER::Layout>::value * smem_buffer_id,
				a_dmem_ptr,
				lda,
				blockIdx_x * SMEM_M, bk,
				m, k
				);
		b_dmem_loader(
				b_smem_ptr + get_smem_size<SMEM_K, SMEM_N, smem_B_skew, typename B_DMEM_LOADER::Layout>::value * smem_buffer_id,
				b_dmem_ptr,
				ldb,
				bk, blockIdx_y * SMEM_N,
				k, n
				);
		cutf::cp_async::wait_group<2>();
		__syncthreads();
		mma_smem<
			T,
			SMEM_M, SMEM_N, SMEM_K,
			FRAG_M, FRAG_N, FRAG_K,
			BLOCK_SIZE,
			typename A_DMEM_LOADER::Layout,
			typename B_DMEM_LOADER::Layout,
			TC_T,
			EC>(
					frag_c,
					a_smem_ptr + get_smem_size<SMEM_M, SMEM_K, smem_A_skew, typename A_DMEM_LOADER::Layout>::value * (1 - smem_buffer_id),
					b_smem_ptr + get_smem_size<SMEM_K, SMEM_N, smem_B_skew, typename B_DMEM_LOADER::Layout>::value * (1 - smem_buffer_id)
				 );
		__syncthreads();
	}
	const auto smem_buffer_id = 1 - ((bk / SMEM_K) % 2);
	cutf::cp_async::wait_all();
	__syncthreads();
	mma_smem<
		T,
		SMEM_M, SMEM_N, SMEM_K,
		FRAG_M, FRAG_N, FRAG_K,
		BLOCK_SIZE,
		typename A_DMEM_LOADER::Layout,
		typename B_DMEM_LOADER::Layout,
		TC_T,
		EC>(
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
				frag_c[i],
				SMEM_M + smem_C_skew
				);
	}
	__syncthreads();

	C_DMEM_STORER c_dmem_storer;
	c_dmem_storer(
			c_dmem_ptr, ldc,
			blockIdx_x * SMEM_M, blockIdx_y * SMEM_N,
			m, n,
			smem,
			alpha, beta
			);
}


template <
	class T,
	unsigned SMEM_M,
	unsigned SMEM_N,
	unsigned SMEM_K,
	class OP_A,
	class OP_B
	>
unsigned get_total_smem_size() {
	return sizeof(T) * std::max<unsigned>(
			(SMEM_M + smem_C_skew) * SMEM_N,
			2 * (
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
	class OP_A,
	class OP_B,
	class TC_T,
	class EC
>
cumpsgemm::gemm_kernel_func_t<T> get_kernel_func_ptr() {
	constexpr cumpsgemm::gemm_kernel_func_t<T> func_ptr = &(gemm_kernel<
		T,
		SMEM_M, SMEM_N, SMEM_K,
		FRAG_M, FRAG_N, FRAG_K,
		BLOCK_SIZE,
		NUM_UNROLLINGS,
		dmem_loader<OP_A, T, SMEM_M, SMEM_K, smem_A_skew, BLOCK_SIZE>,
		dmem_loader<OP_B, T, SMEM_K, SMEM_N, smem_B_skew, BLOCK_SIZE>,
		dmem_storer<T, SMEM_M, SMEM_N, smem_C_skew, BLOCK_SIZE>,
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
	class OP_A,
	class OP_B,
	class TC_T,
	class EC
>
cumpsgemm::gemm_stridedBatch_kernel_func_t<T> get_stridedBatch_kernel_func_ptr() {
	constexpr cumpsgemm::gemm_stridedBatch_kernel_func_t<T> func_ptr = &(gemm_batchStrided_kernel<
		T,
		SMEM_M, SMEM_N, SMEM_K,
		FRAG_M, FRAG_N, FRAG_K,
		BLOCK_SIZE,
		NUM_UNROLLINGS,
		dmem_loader<OP_A, T, SMEM_M, SMEM_K, smem_A_skew, BLOCK_SIZE>,
		dmem_loader<OP_B, T, SMEM_K, SMEM_N, smem_B_skew, BLOCK_SIZE>,
		dmem_storer<T, SMEM_M, SMEM_N, smem_C_skew, BLOCK_SIZE>,
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
	class OP_A,
	class OP_B,
	class TC_T,
	class EC
>
cumpsgemm::gemm_module generate_gemm_module() {
	const auto kernel_func = get_kernel_func_ptr<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, NUM_UNROLLINGS, OP_A, OP_B, TC_T, EC>();
	cumpsgemm::gemm_module mod;
	mod.kernel_func = reinterpret_cast<void*>(kernel_func);
	mod.block_size = BLOCK_SIZE;
	mod.smem_size = get_total_smem_size<T, SMEM_M, SMEM_N, SMEM_K, OP_A, OP_B>();
	mod.smem_m = SMEM_M;
	mod.smem_n = SMEM_N;
	mod.smem_k = SMEM_K;
	CUTF_CHECK_ERROR(cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, mod.smem_size));

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
	class OP_A,
	class OP_B,
	class TC_T,
	class EC
>
cumpsgemm::gemm_module generate_gemm_stridedBatch_module() {
	const auto kernel_func = get_stridedBatch_kernel_func_ptr<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, NUM_UNROLLINGS, OP_A, OP_B, TC_T, EC>();
	cumpsgemm::gemm_module mod;
	mod.kernel_func = reinterpret_cast<void*>(kernel_func);
	mod.block_size = BLOCK_SIZE;
	mod.smem_size = get_total_smem_size<T, SMEM_M, SMEM_N, SMEM_K, OP_A, OP_B>();
	mod.smem_m = SMEM_M;
	mod.smem_n = SMEM_N;
	mod.smem_k = SMEM_K;
	CUTF_CHECK_ERROR(cudaFuncSetAttribute(kernel_func, cudaFuncAttributeMaxDynamicSharedMemorySize, mod.smem_size));

	return mod;
}
} // noname namespace
#endif
