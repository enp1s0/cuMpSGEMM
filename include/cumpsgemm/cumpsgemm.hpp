#ifndef __CUMPSGEMM_HPP__
#define __CUMPSGEMM_HPP__
#include <vector>
#include "cumpsgemm.h"

namespace cumpsgemm {
template <class T>
cublasStatus_t gemm(
		cuMpSGEMM_handle_t handle,
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const uint64_t m,
		const uint64_t n,
		const uint64_t k,
		const T* alpha,
		const T* const a_dmem_ptr, const uint64_t lda,
		const T* const b_dmem_ptr, const uint64_t ldb,
		const T* beta,
		T* const c_dmem_ptr, const uint64_t ldc,
		const cuMpSGEMM_compute_mode_t compute_mode,
		unsigned* const used_kernel_module_id = nullptr
		);

template <class T>
cublasStatus_t gemm_stridedBatch(
		cuMpSGEMM_handle_t handle,
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const uint64_t m,
		const uint64_t n,
		const uint64_t k,
		const T* alpha,
		const T* const a_dmem_ptr, const uint64_t lda, const uint64_t stridea,
		const T* const b_dmem_ptr, const uint64_t ldb, const uint64_t strideb,
		const T* beta,
		T* const c_dmem_ptr, const uint64_t ldc, const uint64_t stridec,
		const uint64_t batch_count,
		const cuMpSGEMM_compute_mode_t compute_mode,
		unsigned* const used_kernel_module_id = nullptr
		);

std::pair<std::size_t, std::size_t> get_exp_stats(
		cuMpSGEMM_handle_t handle,
		const unsigned buffer_id
		);

unsigned get_current_buffer_id(
		cuMpSGEMM_handle_t handle
		);

void set_exp_stats_params(
		cuMpSGEMM_handle_t handle,
		const float ignore_threshold,
		const float lost_threshold
		);

void enable_exp_stats(
		cuMpSGEMM_handle_t handle
		);

void disable_exp_stats(
		cuMpSGEMM_handle_t handle
		);
} // namespace cumpsgemm
#endif
