#ifndef __CUMPGEMM_INTERNAL_HPP__
#define __CUMPGEMM_INTERNAL_HPP__
#include "device_common.hpp"
#include "device_tcec_wrapper.hpp"
namespace cumpsgemm {
// prototype declaration
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
void launch_stridedBatch_kernel (
			const std::size_t m,
			const std::size_t n,
			const std::size_t k,
			const T alpha,
			const T* const a_ptr, const std::size_t lda, const uint64_t stridea,
			const T* const b_ptr, const std::size_t ldb, const uint64_t strideb,
			const T beta,
			T* const c_ptr, const std::size_t ldc, const uint64_t stridec,
			const uint64_t batch_count,
			cudaStream_t cuda_stream
		);

// GEMM Params
constexpr unsigned SMEM_M = 64;
constexpr unsigned SMEM_N = 64;
constexpr unsigned SMEM_K = 32;
constexpr unsigned FRAG_M = 32;
constexpr unsigned FRAG_N = 32;
constexpr unsigned FRAG_K = 32;
constexpr unsigned BLOCK_SIZE = 128;
} // namespace cumpsgemm
#endif
