#ifndef __CUMPSGEMM_HPP__
#define __CUMPSGEMM_HPP__
#include "cumpsgemm.h"
#include <string>
#include <unordered_map>
#include <vector>

namespace cumpsgemm {
using handle_t = cuMpSGEMM_handle_t;
inline void create(handle_t &handle) { cuMpSGEMM_create(&handle); }
inline void destroy(handle_t handle) { cuMpSGEMM_destroy(handle); }
inline void set_stream(handle_t &handle, cudaStream_t cuda_stream) {
  cuMpSGEMM_set_stream(handle, cuda_stream);
}

template <class T>
cublasStatus_t gemm(cuMpSGEMM_handle_t handle, const cublasOperation_t op_A,
                    const cublasOperation_t op_B, const uint64_t m,
                    const uint64_t n, const uint64_t k, const T *alpha,
                    const T *const a_dmem_ptr, const uint64_t lda,
                    const T *const b_dmem_ptr, const uint64_t ldb,
                    const T *beta, T *const c_dmem_ptr, const uint64_t ldc,
                    const cuMpSGEMM_compute_mode_t compute_mode,
                    unsigned *const used_kernel_module_id = nullptr);

template <class T>
cublasStatus_t gemm_stridedBatch(
    cuMpSGEMM_handle_t handle, const cublasOperation_t op_A,
    const cublasOperation_t op_B, const uint64_t m, const uint64_t n,
    const uint64_t k, const T *alpha, const T *const a_dmem_ptr,
    const uint64_t lda, const uint64_t stridea, const T *const b_dmem_ptr,
    const uint64_t ldb, const uint64_t strideb, const T *beta,
    T *const c_dmem_ptr, const uint64_t ldc, const uint64_t stridec,
    const uint64_t batch_count, const cuMpSGEMM_compute_mode_t compute_mode,
    unsigned *const used_kernel_module_id = nullptr);

template <class T>
unsigned exp_stats_ext(cuMpSGEMM_handle_t handle, const unsigned m,
                       const unsigned n, const T *const ptr, const unsigned ld,
                       const unsigned batch_size = 1,
                       const unsigned stride = 0);

template <class T>
unsigned exp_max_ext(cuMpSGEMM_handle_t handle, const unsigned m,
                     const unsigned n, const T *const ptr, const unsigned ld,
                     const unsigned batch_size = 1, const unsigned stride = 0);

std::pair<std::size_t, std::size_t> get_exp_stats(cuMpSGEMM_handle_t handle,
                                                  const unsigned buffer_id);

unsigned get_current_exp_stats_buffer_id(cuMpSGEMM_handle_t handle);

void reset_exp_stats_buffer_id(cuMpSGEMM_handle_t handle);

void set_exp_stats_params(cuMpSGEMM_handle_t handle,
                          const float ignore_threshold,
                          const float underflow_threshold,
                          const float underflow_ratio_tolerance);

void enable_exp_stats(cuMpSGEMM_handle_t handle);

void disable_exp_stats(cuMpSGEMM_handle_t handle);

float get_max_exp(cuMpSGEMM_handle_t handle, const unsigned buffer_id);

cuMpSGEMM_compute_mode_t
get_exp_stats_compute_mode_level(cuMpSGEMM_handle_t handle,
                                 const unsigned buffer_id);

// dynamic scaling
template <class T>
void scale_A(cuMpSGEMM_handle_t handle, const unsigned exp_stats_buffer_id,
             const unsigned dynamic_launch_flag_buffer_id, const unsigned m,
             const unsigned n, T *const ptr, const unsigned ld,
             const unsigned batch_size = 1, const unsigned stride = 0);
template <class T>
void scale_B(cuMpSGEMM_handle_t handle, const unsigned exp_stats_buffer_id,
             const unsigned dynamic_launch_flag_buffer_id, const unsigned m,
             const unsigned n, T *const ptr, const unsigned ld,
             const unsigned batch_size = 1, const unsigned stride = 0);
template <class T>
void scale_C(cuMpSGEMM_handle_t handle, const unsigned exp_stats_buffer_A_id,
             const unsigned exp_stats_buffer_B_id,
             const unsigned dynamic_launch_flag_buffer_id, const unsigned m,
             const unsigned n, T *const ptr, const unsigned ld,
             const unsigned batch_size = 1, const unsigned stride = 0);
template <class T>
void reset_scale_A(cuMpSGEMM_handle_t handle,
                   const unsigned exp_stats_buffer_id,
                   const unsigned dynamic_launch_flag_buffer_id,
                   const unsigned m, const unsigned n, T *const ptr,
                   const unsigned ld, const unsigned batch_size = 1,
                   const unsigned stride = 0);
template <class T>
void reset_scale_B(cuMpSGEMM_handle_t handle,
                   const unsigned exp_stats_buffer_id,
                   const unsigned dynamic_launch_flag_buffer_id,
                   const unsigned m, const unsigned n, T *const ptr,
                   const unsigned ld, const unsigned batch_size = 1,
                   const unsigned stride = 0);
float get_max_exp(cuMpSGEMM_handle_t handle,
                  const unsigned dynamic_launch_flag_buffer_id);

unsigned get_current_dynamic_launch_buffer_id(cuMpSGEMM_handle_t handle);
unsigned get_next_dynamic_launch_buffer_id(cuMpSGEMM_handle_t handle);
cuMpSGEMM_compute_mode_t
get_dynamic_launch_gemm_compute_mode(cuMpSGEMM_handle_t handle,
                                     const unsigned buffer_id);
std::pair<int, int>
get_dynamic_launch_scaling_mode_AB(cuMpSGEMM_handle_t handle,
                                   const unsigned buffer_id);
void set_dynamic_launch_buffer_by_exp_stats(
    cuMpSGEMM_handle *handle, const unsigned dynamic_launch_buffer_id,
    const unsigned A_exp_stats_buffer_id, const unsigned B_exp_stats_buffer_id);

void enable_exp_stats_profiling(cuMpSGEMM_handle *const handle);
void disable_exp_stats_profiling(cuMpSGEMM_handle *const handle);
void reset_exp_stats_profiling(cuMpSGEMM_handle *const handle);
void print_exp_stats_profiling(cuMpSGEMM_handle *const handle,
                               unsigned csv = false);

namespace debug {
struct stats_t {
  double time_sum;
  std::uint64_t n;
};
std::unordered_map<std::string, stats_t>
get_exp_stats_profiling_result(cuMpSGEMM_handle *const handle);
} // namespace debug
} // namespace cumpsgemm
#endif
