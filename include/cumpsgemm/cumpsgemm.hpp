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

template <class T>
unsigned exp_stats_ext(
		cuMpSGEMM_handle_t handle,
		const unsigned m,
		const unsigned n,
		const T* const ptr,
		const unsigned ld,
		const unsigned batch_size = 1,
		const unsigned stride = 0
		);

void download_exp_stats_result(
		cuMpSGEMM_handle_t handle,
		const unsigned buffer_id
		);

std::pair<std::size_t, std::size_t> get_exp_stats(
		cuMpSGEMM_handle_t handle,
		const unsigned buffer_id
		);

unsigned get_current_exp_stats_buffer_id(
		cuMpSGEMM_handle_t handle
		);

void reset_exp_stats_buffer_id(
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

float get_max_exp(
		cuMpSGEMM_handle_t handle,
		const unsigned buffer_id
		);

// dynamic scaling
template <class T>
void scale_AB(
		cuMpSGEMM_handle_t handle,
		const unsigned exp_stats_buffer_id,
		const unsigned dynamic_launch_flag_buffer_id,
		const unsigned m,
		const unsigned n,
		T* const ptr,
		const unsigned ld,
		const unsigned batch_size = 1,
		const unsigned stride = 0
		);
template <class T>
void scale_C(
		cuMpSGEMM_handle_t handle,
		const unsigned exp_stats_buffer_A_id,
		const unsigned exp_stats_buffer_B_id,
		const unsigned dynamic_launch_flag_buffer_id,
		const unsigned m,
		const unsigned n,
		T* const ptr,
		const unsigned ld,
		const unsigned batch_size = 1,
		const unsigned stride = 0
		);
template <class T>
void reset_scale_AB(
		cuMpSGEMM_handle_t handle,
		const unsigned exp_stats_buffer_id,
		const unsigned dynamic_launch_flag_buffer_id,
		const unsigned m,
		const unsigned n,
		T* const ptr,
		const unsigned ld,
		const unsigned batch_size = 1,
		const unsigned stride = 0
		);
float get_max_exp(
		cuMpSGEMM_handle_t handle,
		const unsigned dynamic_launch_flag_buffer_id
		);
} // namespace cumpsgemm
#endif
