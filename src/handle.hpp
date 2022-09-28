#pragma once
#include <cstdint>
#include <cuComplex.h>

namespace cumpsgemm {
namespace device {
template <class T>
struct element_t_conv {using type = T;};
template <> struct element_t_conv<float2 > {using type = float;};
} // namespace device
// for exp stats
using counter_t = unsigned long long int;

template <class T>
using gemm_kernel_func_t = void (*)(
			const uint32_t,
			const uint32_t,
			const uint32_t,
			const T,
			const T* const, const uint32_t,
			const T* const, const uint32_t,
			const T,
			T* const, const uint32_t,
			const typename device::element_t_conv<T>::type,
			const typename device::element_t_conv<T>::type,
			counter_t* const,
			counter_t* const
			);

template <class T>
using gemm_stridedBatch_kernel_func_t = void (*)(
			const uint32_t,
			const uint32_t,
			const uint32_t,
			const T,
			const T* const, const uint32_t, const uint64_t,
			const T* const, const uint32_t, const uint64_t,
			const T,
			T* const, const uint32_t, const uint64_t,
			const uint32_t,
			const typename device::element_t_conv<T>::type,
			const typename device::element_t_conv<T>::type,
			counter_t* const,
			counter_t* const
			);

struct gemm_module {
	void* kernel_func;

	unsigned smem_m, smem_n, smem_k;
	unsigned smem_size;
	unsigned block_size;
	unsigned num_active_blocks;
};

namespace kernel_module_code {
using code_t = std::uint32_t;
constexpr code_t op_a_col_major   = 0b0'0'0'00'01;
constexpr code_t op_a_row_major   = 0b0'0'0'00'10;
constexpr code_t op_a_conjugate   = 0b0'0'0'00'11;
constexpr code_t op_b_col_major   = 0b0'0'0'01'00;
constexpr code_t op_b_row_major   = 0b0'0'0'10'00;
constexpr code_t op_b_conjugate   = 0b0'0'0'11'00;
constexpr code_t half             = 0b0'0'0'00'00;
constexpr code_t tf32             = 0b0'0'1'00'00;
constexpr code_t with_ec          = 0b0'0'0'00'00;
constexpr code_t without_ec       = 0b0'1'0'00'00;
constexpr code_t s                = 0b0'0'0'00'00;
constexpr code_t c                = 0b1'0'0'00'00;
// ------- OR accumulation ------
constexpr code_t max_code = 0b1'11'11'11 + 1;
} // namespace kernel_module_code
} // namespace cumpsgemm

struct cuMpSGEMM_handle {
	unsigned num_sms;
	// 0 is for large size matmul and (num_kernel_candidates - 1) is for small.
	static constexpr unsigned num_kernel_candidates = 3;

	cumpsgemm::gemm_module gemm_module             [cumpsgemm::kernel_module_code::max_code][num_kernel_candidates];
	cumpsgemm::gemm_module gemm_stridedBatch_module[cumpsgemm::kernel_module_code::max_code][num_kernel_candidates];

	// cuda stream
	cudaStream_t cuda_stream = 0;

	// For exp stats
	cumpsgemm::counter_t* dev_total_counter;
	cumpsgemm::counter_t* dev_lost_counter;
	cumpsgemm::counter_t* host_total_counter;
	cumpsgemm::counter_t* host_lost_counter;

	float ignore_threshold;
	float lost_threshold;

	bool exp_stats_enabled;
	std::uint32_t counter_length;
	std::uint32_t counter_offset;
	std::uint32_t last_stored_counter_length;
};

namespace cumpsgemm {
namespace exp_stats {
// exp_stats API
void resize_counter(
		cuMpSGEMM_handle* handle,
		const std::size_t new_length
		);
void init_counter (
		cuMpSGEMM_handle* handle,
		const unsigned length
		);
void exp_stats_ext(
		cuMpSGEMM_handle* handle,
		const unsigned m,
		const unsigned n,
		const float* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride
		);
void exp_stats_ext(
		cuMpSGEMM_handle* handle,
		const unsigned m,
		const unsigned n,
		const cuComplex* const ptr,
		const unsigned ld,
		const unsigned batch_size,
		const unsigned stride
		);
} // namespace exp_stats
} // namespace cumpsgemm
