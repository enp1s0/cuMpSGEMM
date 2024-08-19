#pragma once
#include "instance.hpp"
#include <cstdint>
#include <cuComplex.h>
#include <utility>

struct cuMpSGEMM_handle {
  unsigned num_sms;

  cumpsgemm::gemm_module gemm_module[cumpsgemm::kernel_module_code::max_code]
                                    [cumpsgemm::num_kernel_candidates];
  cumpsgemm::gemm_module
      gemm_stridedBatch_module[cumpsgemm::kernel_module_code::max_code]
                              [cumpsgemm::num_kernel_candidates];
  cumpsgemm::gemm_module
      gemm_atomic_module[cumpsgemm::kernel_module_code::max_code];

  // cuda stream
  cudaStream_t cuda_stream = 0;

  // For exp stats
  cumpsgemm::exp_stats::exp_stats_handle *exp_stats_handle;

  // For dynamic launch
  cumpsgemm::dynamic_launch::dynamic_launch_handle *dynamic_launch_handle;

  float *temp_working_memory;
  std::size_t temp_working_memory_float_count;
};

void init_exp_stats_counter_buffer(cuMpSGEMM_handle *handle);
void destroy_exp_stats_counter_buffer(cuMpSGEMM_handle *handle);
void init_temp_working_memory(cuMpSGEMM_handle *handle);

void init_dynamic_launch_flag_buffer(cuMpSGEMM_handle *handle);
void destroy_launch_flag_buffer(cuMpSGEMM_handle *handle);
void destroy_temp_working_memory(cuMpSGEMM_handle *handle);
