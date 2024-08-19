#pragma once
#include "handle.hpp"
#include <cutf/debug/time_breakdown.hpp>

namespace cumpsgemm {
namespace exp_stats {
struct exp_stats_handle {
  bool enabled;

  cumpsgemm::counter_t *dev_total_count_buffer;
  cumpsgemm::counter_t *dev_underflow_count_buffer;
  int *dev_compute_mode_buffer;
  float *dev_max_abs_buffer;

  ulong2 *host_counter_buffer;
  static constexpr cumpsgemm::counter_t buffer_empty_value = ~0llu;

  float ignore_threshold;
  float underflow_threshold;
  float underflow_tolerance_rate;

  std::uint32_t buffer_length;
  std::uint32_t current_buffer_id;
  bool counter_init_disabled;

  // For profiling
  cutf::debug::time_breakdown::profiler profiler;
  int profiling_enabled = false;
};
// exp_stats API
void resize_counter(cuMpSGEMM_handle *handle, const std::size_t new_length);
void init_counter(cuMpSGEMM_handle *handle, const unsigned buffer_id);
std::uint32_t get_next_exp_stats_buffer_id(cuMpSGEMM_handle *handle);
std::uint32_t get_current_exp_stats_buffer_id(cuMpSGEMM_handle *handle);
void reset_exp_stats_buffer_id(cuMpSGEMM_handle *handle);
void download_exp_stats(cuMpSGEMM_handle *handle, const unsigned buffer_id);
std::pair<std::size_t, std::size_t> get_exp_stats(cuMpSGEMM_handle *handle,
                                                  const unsigned buffer_id);
template <class T>
void exp_stats_ext(cuMpSGEMM_handle *handle, const unsigned m, const unsigned n,
                   const T *const ptr, const unsigned ld,
                   const unsigned batch_size, const unsigned stride);
template <class T>
void exp_max_ext(cuMpSGEMM_handle *handle, const unsigned m, const unsigned n,
                 const T *const ptr, const unsigned ld,
                 const unsigned batch_size, const unsigned stride);
cuMpSGEMM_compute_mode_t get_compute_mode_level(cuMpSGEMM_handle *handle,
                                                const unsigned buffer_id);
} // namespace exp_stats
} // namespace cumpsgemm
