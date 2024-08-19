#pragma once
#include "handle.hpp"

namespace cumpsgemm {
namespace dynamic_scaling {
template <class T>
void scale_A(cuMpSGEMM_handle *handle, const unsigned m, const unsigned n,
             T *const ptr, const unsigned ld, const unsigned stride,
             const unsigned batch_size, const unsigned exp_stats_buffer_id,
             const unsigned dynamic_launch_buffer_id);

template <class T>
void scale_B(cuMpSGEMM_handle *handle, const unsigned m, const unsigned n,
             T *const ptr, const unsigned ld, const unsigned stride,
             const unsigned batch_size, const unsigned exp_stats_buffer_id,
             const unsigned dynamic_launch_buffer_id);

template <class T>
void scale_C(cuMpSGEMM_handle *handle, const unsigned m, const unsigned n,
             T *const ptr, const unsigned ld, const unsigned stride,
             const unsigned batch_size, const unsigned exp_stats_buffer_A_id,
             const unsigned exp_stats_buffer_B_id,
             const unsigned dynamic_launch_buffer_id);

float get_max_exp(cuMpSGEMM_handle *handle, const unsigned exp_stats_buffer_id);

template <class T>
void reset_scale_A(cuMpSGEMM_handle *handle, const unsigned m, const unsigned n,
                   T *const ptr, const unsigned ld, const unsigned stride,
                   const unsigned batch_size,
                   const unsigned exp_stats_buffer_id,
                   const unsigned dynamic_launch_buffer_id);

template <class T>
void reset_scale_B(cuMpSGEMM_handle *handle, const unsigned m, const unsigned n,
                   T *const ptr, const unsigned ld, const unsigned stride,
                   const unsigned batch_size,
                   const unsigned exp_stats_buffer_id,
                   const unsigned dynamic_launch_buffer_id);

void set_dynamic_launch_buffer_by_exp_stats(
    cuMpSGEMM_handle *handle, const unsigned dynamic_mode_flag_id,
    const unsigned A_exp_stats_buffer_id, const unsigned B_exp_stats_buffer_id);
} // namespace dynamic_scaling
} // namespace cumpsgemm
