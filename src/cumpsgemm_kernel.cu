#include <cassert>
#include <type_traits>
#include <wmma_extension/utils.hpp>
#include <cutf/cuda.hpp>
#include <cutf/error.hpp>

#include <cumpsgemm/cumpsgemm.h>
#include "device_tcec_wrapper.hpp"

namespace {
constexpr unsigned smem_A_skew = 8;
constexpr unsigned smem_B_skew = 8;
constexpr unsigned smem_C_skew = 8;
constexpr unsigned warp_size = 32;

// smem size
template <unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, class Layout>
struct get_smem_size                                             {static constexpr unsigned value = (SMEM_M + SKEW) * SMEM_N;};
template <unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW>
struct get_smem_size<SMEM_M, SMEM_N, SKEW, cumpsgemm::row_major> {static constexpr unsigned value = (SMEM_N + SKEW) * SMEM_M;};

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
			for (unsigned offset = 0; offset < SMEM_M * SMEM_N; offset += BLOCK_SIZE) {
				const auto index = offset + threadIdx.x;
				const auto m = index % SMEM_M;
				const auto n = index / SMEM_M;
				const auto smem_index = m + n * (SMEM_M + SKEW);
				const auto dmem_index = (start_m + m) + (start_n + n) * ld;
				smem_ptr[smem_index] = dmem_ptr[dmem_index];
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
struct dmem_loader<cumpsgemm::conjugate, cuComplex, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE> {
	using Layout = cumpsgemm::col_major;
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
	class A_DMEM_LOADER,
	class B_DMEM_LOADER,
	class C_DMEM_STORER,
	class TC_T,
	class EC
>
__global__ void gemm_kernel(
		const uint64_t m,
		const uint64_t n,
		const uint64_t k,
		const T alpha,
		const T* const a_dmem_ptr, const uint64_t lda,
		const T* const b_dmem_ptr, const uint64_t ldb,
		const T beta,
		T* const c_dmem_ptr, const uint64_t ldc
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

	unsigned smem_buffer_id = 0;
	a_dmem_loader(
			a_smem_ptr + get_smem_size<SMEM_M, SMEM_K, smem_A_skew, typename A_DMEM_LOADER::Layout>::value * smem_buffer_id,
			a_dmem_ptr,
			lda,
			blockIdx.x * SMEM_M, 0,
			m, k
			);
	b_dmem_loader(
			b_smem_ptr + get_smem_size<SMEM_K, SMEM_N, smem_B_skew, typename B_DMEM_LOADER::Layout>::value * smem_buffer_id,
			b_dmem_ptr,
			ldb,
			0, blockIdx.y * SMEM_N,
			k, n
			);
	__syncthreads();
	for (uint64_t bk = SMEM_K; bk < k; bk += SMEM_K) {
		smem_buffer_id = (bk / SMEM_K) % 2;
		a_dmem_loader(
				a_smem_ptr + get_smem_size<SMEM_M, SMEM_K, smem_A_skew, typename A_DMEM_LOADER::Layout>::value * smem_buffer_id,
				a_dmem_ptr,
				lda,
				blockIdx.x * SMEM_M, bk,
				m, k
				);
		b_dmem_loader(
				b_smem_ptr + get_smem_size<SMEM_K, SMEM_N, smem_B_skew, typename B_DMEM_LOADER::Layout>::value * smem_buffer_id,
				b_dmem_ptr,
				ldb,
				bk, blockIdx.y * SMEM_N,
				k, n
				);
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
			blockIdx.x * SMEM_M, blockIdx.y * SMEM_N,
			m, n,
			smem,
			alpha, beta
			);
}

template <class T>
using kernel_func_t = void (*)(
			const uint64_t,
			const uint64_t,
			const uint64_t,
			const T,
			const T* const, const uint64_t,
			const T* const, const uint64_t,
			const T,
			T* const, const uint64_t
			);


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
kernel_func_t<T> get_kernel_func_ptr() {
	constexpr kernel_func_t<T> func_ptr = &(gemm_kernel<
		T,
		SMEM_M, SMEM_N, SMEM_K,
		FRAG_M, FRAG_N, FRAG_K,
		BLOCK_SIZE,
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
	class OP_A,
	class OP_B,
	class TC_T,
	class EC
>
void launch_kernel (
			const std::size_t m,
			const std::size_t n,
			const std::size_t k,
			const T alpha,
			const T* const a_ptr, const std::size_t lda,
			const T* const b_ptr, const std::size_t ldb,
			const T beta,
			T* const c_ptr, const std::size_t ldc,
			cudaStream_t cuda_stream
		) {
	const auto smem_size_in_byte = get_total_smem_size<
		T,
		SMEM_M, SMEM_N, SMEM_K,
		OP_A, OP_B>();
	const auto kernel_ptr = get_kernel_func_ptr<
		T,
		SMEM_M, SMEM_N, SMEM_K,
		FRAG_M, FRAG_N, FRAG_K,
		BLOCK_SIZE,
		OP_A, OP_B,
		TC_T, EC>();
	CUTF_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_in_byte));

	const dim3 block_size(BLOCK_SIZE);
	const dim3 grid_size(
			(m + SMEM_M - 1) / SMEM_M,
			(n + SMEM_N - 1) / SMEM_N
			);

	kernel_ptr<<<grid_size, block_size, smem_size_in_byte, cuda_stream>>>(
			m, n, k,
			alpha,
			a_ptr, lda,
			b_ptr, ldb,
			beta,
			c_ptr, ldc
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
	class TC_T,
	class EC
>
void layout_selector (
			const cublasOperation_t op_A,
			const cublasOperation_t op_B,
			const std::size_t m,
			const std::size_t n,
			const std::size_t k,
			const T alpha,
			const T* const a_ptr, const std::size_t lda,
			const T* const b_ptr, const std::size_t ldb,
			const T beta,
			T* const c_ptr, const std::size_t ldc,
			cudaStream_t cuda_stream
		) {
	assert((std::is_same<T, float>::value && !(op_A == CUBLAS_OP_C || op_B == CUBLAS_OP_C)));
#define CASE(A, a, B, b) \
	if (op_A == a && op_B == b) {launch_kernel<T, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, A, B, TC_T, EC>(m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta, c_ptr, ldc, cuda_stream);return;}

	CASE(cumpsgemm::col_major, CUBLAS_OP_N, cumpsgemm::col_major, CUBLAS_OP_N);
	CASE(cumpsgemm::row_major, CUBLAS_OP_T, cumpsgemm::col_major, CUBLAS_OP_N);
	CASE(cumpsgemm::conjugate, CUBLAS_OP_C, cumpsgemm::col_major, CUBLAS_OP_N);
	CASE(cumpsgemm::col_major, CUBLAS_OP_N, cumpsgemm::row_major, CUBLAS_OP_T);
	CASE(cumpsgemm::row_major, CUBLAS_OP_T, cumpsgemm::row_major, CUBLAS_OP_T);
	CASE(cumpsgemm::conjugate, CUBLAS_OP_C, cumpsgemm::row_major, CUBLAS_OP_T);
	CASE(cumpsgemm::col_major, CUBLAS_OP_N, cumpsgemm::conjugate, CUBLAS_OP_C);
	CASE(cumpsgemm::row_major, CUBLAS_OP_T, cumpsgemm::conjugate, CUBLAS_OP_C);
	CASE(cumpsgemm::conjugate, CUBLAS_OP_C, cumpsgemm::conjugate, CUBLAS_OP_C);
}
} // noname namespace

extern "C" cublasStatus_t cuMpSGEMM_sgemm(
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const uint64_t m,
		const uint64_t n,
		const uint64_t k,
		const float* alpha,
		const float* const a_dmem_ptr, const uint64_t lda,
		const float* const b_dmem_ptr, const uint64_t ldb,
		const float* beta,
		float* const c_dmem_ptr, const uint64_t ldc,
		const cuMpSGEMM_compute_mode_t compute_mode,
		cudaStream_t cuda_stream
		) {
	constexpr unsigned SMEM_M = 64;
	constexpr unsigned SMEM_N = 64;
	constexpr unsigned SMEM_K = 64;
	constexpr unsigned FRAG_M = 32;
	constexpr unsigned FRAG_N = 32;
	constexpr unsigned FRAG_K = 32;
	constexpr unsigned BLOCK_SIZE = 128;
	switch (compute_mode) {
	case CUMPSGEMM_FP16TC:   layout_selector<float, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, half                         , mtk::wmma::tcec::without_ec>(op_A, op_B, m, n, k, *alpha, a_dmem_ptr, lda, b_dmem_ptr, ldb, *beta, c_dmem_ptr, ldc, cuda_stream);break;
	case CUMPSGEMM_FP16TCEC: layout_selector<float, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, half                         , mtk::wmma::tcec::with_ec   >(op_A, op_B, m, n, k, *alpha, a_dmem_ptr, lda, b_dmem_ptr, ldb, *beta, c_dmem_ptr, ldc, cuda_stream);break;
	case CUMPSGEMM_TF32TC:   layout_selector<float, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, nvcuda::wmma::precision::tf32, mtk::wmma::tcec::without_ec>(op_A, op_B, m, n, k, *alpha, a_dmem_ptr, lda, b_dmem_ptr, ldb, *beta, c_dmem_ptr, ldc, cuda_stream);break;
	case CUMPSGEMM_TF32TCEC: layout_selector<float, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, nvcuda::wmma::precision::tf32, mtk::wmma::tcec::with_ec   >(op_A, op_B, m, n, k, *alpha, a_dmem_ptr, lda, b_dmem_ptr, ldb, *beta, c_dmem_ptr, ldc, cuda_stream);break;
	default:break;
	}

	return CUBLAS_STATUS_SUCCESS;
}

extern "C" cublasStatus_t cuMpSGEMM_cgemm(
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const uint64_t m,
		const uint64_t n,
		const uint64_t k,
		const cuComplex* alpha,
		const cuComplex* const a_dmem_ptr, const uint64_t lda,
		const cuComplex* const b_dmem_ptr, const uint64_t ldb,
		const cuComplex* beta,
		cuComplex* const c_dmem_ptr, const uint64_t ldc,
		const cuMpSGEMM_compute_mode_t compute_mode,
		cudaStream_t cuda_stream
		) {
	constexpr unsigned SMEM_M = 64;
	constexpr unsigned SMEM_N = 64;
	constexpr unsigned SMEM_K = 64;
	constexpr unsigned FRAG_M = 32;
	constexpr unsigned FRAG_N = 32;
	constexpr unsigned FRAG_K = 32;
	constexpr unsigned BLOCK_SIZE = 128;
	switch (compute_mode) {
	case CUMPSGEMM_FP16TC:   layout_selector<cuComplex, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, half                         , mtk::wmma::tcec::without_ec>(op_A, op_B, m, n, k, *alpha, a_dmem_ptr, lda, b_dmem_ptr, ldb, *beta, c_dmem_ptr, ldc, cuda_stream);break;
	case CUMPSGEMM_FP16TCEC: layout_selector<cuComplex, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, half                         , mtk::wmma::tcec::with_ec   >(op_A, op_B, m, n, k, *alpha, a_dmem_ptr, lda, b_dmem_ptr, ldb, *beta, c_dmem_ptr, ldc, cuda_stream);break;
	case CUMPSGEMM_TF32TC:   layout_selector<cuComplex, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, nvcuda::wmma::precision::tf32, mtk::wmma::tcec::without_ec>(op_A, op_B, m, n, k, *alpha, a_dmem_ptr, lda, b_dmem_ptr, ldb, *beta, c_dmem_ptr, ldc, cuda_stream);break;
	case CUMPSGEMM_TF32TCEC: layout_selector<cuComplex, SMEM_M, SMEM_N, SMEM_K, FRAG_M, FRAG_N, FRAG_K, BLOCK_SIZE, nvcuda::wmma::precision::tf32, mtk::wmma::tcec::with_ec   >(op_A, op_B, m, n, k, *alpha, a_dmem_ptr, lda, b_dmem_ptr, ldb, *beta, c_dmem_ptr, ldc, cuda_stream);break;
	default:break;
	}

	return CUBLAS_STATUS_SUCCESS;
}
