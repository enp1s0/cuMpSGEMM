#include "dynamic_launch.hpp"
#include "dynamic_launch_utils.hpp"
#include "dynamic_scaling.hpp"
#include "exp_stats.hpp"
#include <algorithm>
#include <cumpsgemm/detail/common.h>
#include <cutf/memory.hpp>

namespace {
__device__ float mul_a(const float v, const float a) { return v * a; }
__device__ float2 mul_a(const cuComplex v, const float a) {
  return make_float2(v.x * a, v.y * a);
}

constexpr float half_exp_max = 1u << 14;

enum scaling_matrix_t { matrix_A, matrix_B };

template <unsigned BLOCK_SIZE, unsigned VEC_LEN, class LOOP_T, class T>
__device__ void scaling_core(const unsigned m, const unsigned n, T *const ptr,
                             const unsigned ld, const unsigned batch_size,
                             const unsigned stride, const float coef) {
  const auto ib = blockIdx.y;
  auto local_mat_ptr = ptr + ib * stride;
  for (LOOP_T lid = (threadIdx.x + blockIdx.x * blockDim.x) * VEC_LEN;
       lid < m * n; lid += BLOCK_SIZE * gridDim.x * VEC_LEN) {
    T vec[VEC_LEN];
    if (lid + VEC_LEN < m * n) {
      for (uint32_t i = 0; i < VEC_LEN; i++) {
        const auto gid = lid + i;
        const auto im = gid % m;
        const auto in = gid / m;

        const auto memory_index = im + ld * in;
        vec[i] = local_mat_ptr[memory_index];
      }

      for (uint32_t i = 0; i < VEC_LEN; i++) {
        const auto gid = lid + i;
        const auto im = gid % m;
        const auto in = gid / m;

        const auto memory_index = im + ld * in;
        local_mat_ptr[memory_index] = mul_a(vec[i], coef);
      }
    } else {
      for (uint32_t i = 0; i < VEC_LEN; i++) {
        const auto gid = lid + i;
        if (gid < m * n) {
          const auto im = gid % m;
          const auto in = gid / m;

          const auto memory_index = im + ld * in;
          vec[i] = local_mat_ptr[memory_index];
        } else {
          break;
        }
      }
      for (uint32_t i = 0; i < VEC_LEN; i++) {
        const auto gid = lid + i;
        if (gid < m * n) {
          const auto im = gid % m;
          const auto in = gid / m;

          const auto memory_index = im + ld * in;
          local_mat_ptr[memory_index] = mul_a(vec[i], coef);
        } else {
          break;
        }
      }
    }
  }
}

template <unsigned BLOCK_SIZE, unsigned VEC_LEN, class T, class LOOP_T>
__global__ void scaling_kernel(const int *const dynamic_mode, const unsigned m,
                               const unsigned n, T *const ptr,
                               const unsigned ld, const unsigned batch_size,
                               const unsigned stride,
                               const float *const max_abs_value_ptr,
                               const scaling_matrix_t scaling_matrix) {
  if (dynamic_mode != nullptr) {
    if (scaling_matrix == matrix_A &&
        !cumpsgemm::dynamic_launch::utils::get_scale_A_flag(*dynamic_mode))
      return;
    if (scaling_matrix == matrix_B &&
        !cumpsgemm::dynamic_launch::utils::get_scale_B_flag(*dynamic_mode))
      return;
  }
  if (*max_abs_value_ptr == 0) {
    return;
  }
  const auto coef = half_exp_max / *max_abs_value_ptr;

  scaling_core<BLOCK_SIZE, VEC_LEN, LOOP_T, T>(m, n, ptr, ld, batch_size,
                                               stride, coef);
}

template <unsigned BLOCK_SIZE, unsigned VEC_LEN, class T, class LOOP_T>
__global__ void scaling_kernel(const int *const dynamic_mode, const unsigned m,
                               const unsigned n, T *const ptr,
                               const unsigned ld, const unsigned batch_size,
                               const unsigned stride,
                               const float *const max_abs_value_A_ptr,
                               const float *const max_abs_value_B_ptr) {
  auto coef = 1.f;
  if (*max_abs_value_A_ptr == 0 || *max_abs_value_B_ptr == 0) {
    return;
  }
  if (dynamic_mode != nullptr) {
    const auto mode = *dynamic_mode;
    if ((!cumpsgemm::dynamic_launch::utils::get_scale_A_flag(mode)) &&
        (!cumpsgemm::dynamic_launch::utils::get_scale_B_flag(mode)))
      return;
    if (cumpsgemm::dynamic_launch::utils::get_scale_A_flag(mode))
      coef *= *max_abs_value_A_ptr / half_exp_max;
    if (cumpsgemm::dynamic_launch::utils::get_scale_B_flag(mode))
      coef *= *max_abs_value_B_ptr / half_exp_max;
  } else {
    coef = static_cast<float>(*max_abs_value_A_ptr * *max_abs_value_B_ptr) /
           (half_exp_max * half_exp_max);
  }

  scaling_core<BLOCK_SIZE, VEC_LEN, LOOP_T, T>(m, n, ptr, ld, batch_size,
                                               stride, coef);
}

template <unsigned BLOCK_SIZE, unsigned VEC_LEN, class T, class LOOP_T>
__global__ void
reset_scaling_kernel(const int *const dynamic_mode, const unsigned m,
                     const unsigned n, T *const ptr, const unsigned ld,
                     const unsigned batch_size, const unsigned stride,
                     const float *const max_abs_value_ptr,
                     const scaling_matrix_t scaling_matrix) {
  if (dynamic_mode != nullptr) {
    if (scaling_matrix == matrix_A &&
        !cumpsgemm::dynamic_launch::utils::get_scale_A_flag(*dynamic_mode))
      return;
    if (scaling_matrix == matrix_B &&
        !cumpsgemm::dynamic_launch::utils::get_scale_B_flag(*dynamic_mode))
      return;
  }
  if (*max_abs_value_ptr == 0) {
    return;
  }
  const auto coef = *max_abs_value_ptr / half_exp_max;

  scaling_core<BLOCK_SIZE, VEC_LEN, LOOP_T, T>(m, n, ptr, ld, batch_size,
                                               stride, coef);
}

template <class T>
void scale_AB(cuMpSGEMM_handle *handle, const unsigned m, const unsigned n,
              T *const ptr, const unsigned ld, const unsigned stride,
              const unsigned batch_size, const unsigned exp_stats_buffer_id,
              const unsigned dynamic_launch_buffer_id,
              const scaling_matrix_t scaling_matrix) {
  constexpr unsigned VEC_LEN = 2;

  constexpr auto block_size = 256;
  const dim3 grid_size(
      ((1lu * m * n + block_size - 1) / block_size + VEC_LEN - 1) / VEC_LEN,
      (stride == 0) ? 1 : batch_size);

  if (handle->exp_stats_handle->profiling_enabled) {
    handle->exp_stats_handle->profiler.start_timer_sync("scaling_AB");
  }
  if (static_cast<std::size_t>(m) * n < (1lu << 32)) {
    using LOOP_T = unsigned;
    scaling_kernel<block_size, VEC_LEN, T, LOOP_T>
        <<<grid_size, block_size, 0, handle->cuda_stream>>>(
            handle->dynamic_launch_handle->flag_buffer +
                dynamic_launch_buffer_id,
            m, n, ptr, ld, batch_size, stride,
            handle->exp_stats_handle->dev_max_abs_buffer + exp_stats_buffer_id,
            scaling_matrix);
  } else {
    using LOOP_T = std::size_t;
    scaling_kernel<block_size, VEC_LEN, T, LOOP_T>
        <<<grid_size, block_size, 0, handle->cuda_stream>>>(
            handle->dynamic_launch_handle->flag_buffer +
                dynamic_launch_buffer_id,
            m, n, ptr, ld, batch_size, stride,
            handle->exp_stats_handle->dev_max_abs_buffer + exp_stats_buffer_id,
            scaling_matrix);
  }
  if (handle->exp_stats_handle->profiling_enabled) {
    handle->exp_stats_handle->profiler.stop_timer_sync("scaling_AB");
  }
}

template <class T>
void reset_scale_AB(cuMpSGEMM_handle *handle, const unsigned m,
                    const unsigned n, T *const ptr, const unsigned ld,
                    const unsigned stride, const unsigned batch_size,
                    const unsigned exp_stats_buffer_id,
                    const unsigned dynamic_launch_buffer_id,
                    const scaling_matrix_t scaling_matrix) {
  constexpr unsigned VEC_LEN = 2;

  constexpr auto block_size = 256;
  const dim3 grid_size(
      ((1lu * m * n + block_size - 1) / block_size + VEC_LEN - 1) / VEC_LEN,
      (stride == 0) ? 1 : batch_size);

  if (static_cast<std::size_t>(m) * n < (1lu << 32)) {
    using LOOP_T = unsigned;
    reset_scaling_kernel<block_size, VEC_LEN, T, LOOP_T>
        <<<grid_size, block_size, 0, handle->cuda_stream>>>(
            handle->dynamic_launch_handle->flag_buffer +
                dynamic_launch_buffer_id,
            m, n, ptr, ld, batch_size, stride,
            handle->exp_stats_handle->dev_max_abs_buffer + exp_stats_buffer_id,
            scaling_matrix);
  } else {
    using LOOP_T = std::size_t;
    reset_scaling_kernel<block_size, VEC_LEN, T, LOOP_T>
        <<<grid_size, block_size, 0, handle->cuda_stream>>>(
            handle->dynamic_launch_handle->flag_buffer +
                dynamic_launch_buffer_id,
            m, n, ptr, ld, batch_size, stride,
            handle->exp_stats_handle->dev_max_abs_buffer + exp_stats_buffer_id,
            scaling_matrix);
  }
}
} // unnamed namespace

template <class T>
void cumpsgemm::dynamic_scaling::scale_A(
    cuMpSGEMM_handle *handle, const unsigned m, const unsigned n, T *const ptr,
    const unsigned ld, const unsigned stride, const unsigned batch_size,
    const unsigned exp_stats_buffer_id,
    const unsigned dynamic_launch_buffer_id) {
  scale_AB(handle, m, n, ptr, ld, stride, batch_size, exp_stats_buffer_id,
           dynamic_launch_buffer_id, matrix_A);
}

template <class T>
void cumpsgemm::dynamic_scaling::scale_B(
    cuMpSGEMM_handle *handle, const unsigned m, const unsigned n, T *const ptr,
    const unsigned ld, const unsigned stride, const unsigned batch_size,
    const unsigned exp_stats_buffer_id,
    const unsigned dynamic_launch_buffer_id) {
  scale_AB(handle, m, n, ptr, ld, stride, batch_size, exp_stats_buffer_id,
           dynamic_launch_buffer_id, matrix_B);
}

template void cumpsgemm::dynamic_scaling::scale_A<float>(
    cuMpSGEMM_handle *, const unsigned, const unsigned, float *const,
    const unsigned, const unsigned, const unsigned, const unsigned,
    const unsigned);
template void cumpsgemm::dynamic_scaling::scale_A<cuComplex>(
    cuMpSGEMM_handle *, const unsigned, const unsigned, cuComplex *const,
    const unsigned, const unsigned, const unsigned, const unsigned,
    const unsigned);
template void cumpsgemm::dynamic_scaling::scale_B<float>(
    cuMpSGEMM_handle *, const unsigned, const unsigned, float *const,
    const unsigned, const unsigned, const unsigned, const unsigned,
    const unsigned);
template void cumpsgemm::dynamic_scaling::scale_B<cuComplex>(
    cuMpSGEMM_handle *, const unsigned, const unsigned, cuComplex *const,
    const unsigned, const unsigned, const unsigned, const unsigned,
    const unsigned);

template <class T>
void cumpsgemm::dynamic_scaling::scale_C(
    cuMpSGEMM_handle *handle, const unsigned m, const unsigned n, T *const ptr,
    const unsigned ld, const unsigned stride, const unsigned batch_size,
    const unsigned exp_stats_buffer_A_id, const unsigned exp_stats_buffer_B_id,
    const unsigned dynamic_launch_buffer_id) {
  constexpr unsigned VEC_LEN = 2;

  constexpr auto block_size = 256;
  const dim3 grid_size(
      ((1lu * m * n + block_size - 1) / block_size + VEC_LEN - 1) / VEC_LEN,
      (stride == 0) ? 1 : batch_size);

  if (handle->exp_stats_handle->profiling_enabled) {
    handle->exp_stats_handle->profiler.start_timer_sync("scaling_C");
  }
  if (static_cast<std::size_t>(m) * n < (1lu << 32)) {
    using LOOP_T = unsigned;
    scaling_kernel<block_size, VEC_LEN, T, LOOP_T>
        <<<grid_size, block_size, 0, handle->cuda_stream>>>(
            handle->dynamic_launch_handle->flag_buffer +
                dynamic_launch_buffer_id,
            m, n, ptr, ld, batch_size, stride,
            handle->exp_stats_handle->dev_max_abs_buffer +
                exp_stats_buffer_A_id,
            handle->exp_stats_handle->dev_max_abs_buffer +
                exp_stats_buffer_B_id);
  } else {
    using LOOP_T = std::size_t;
    scaling_kernel<block_size, VEC_LEN, T, LOOP_T>
        <<<grid_size, block_size, 0, handle->cuda_stream>>>(
            handle->dynamic_launch_handle->flag_buffer +
                dynamic_launch_buffer_id,
            m, n, ptr, ld, batch_size, stride,
            handle->exp_stats_handle->dev_max_abs_buffer +
                exp_stats_buffer_A_id,
            handle->exp_stats_handle->dev_max_abs_buffer +
                exp_stats_buffer_B_id);
  }
  if (handle->exp_stats_handle->profiling_enabled) {
    handle->exp_stats_handle->profiler.stop_timer_sync("scaling_C");
  }
}
template void cumpsgemm::dynamic_scaling::scale_C<float>(
    cuMpSGEMM_handle *, const unsigned, const unsigned, float *const,
    const unsigned, const unsigned, const unsigned, const unsigned,
    const unsigned, const unsigned);
template void cumpsgemm::dynamic_scaling::scale_C<cuComplex>(
    cuMpSGEMM_handle *, const unsigned, const unsigned, cuComplex *const,
    const unsigned, const unsigned, const unsigned, const unsigned,
    const unsigned, const unsigned);

template <class T>
void cumpsgemm::dynamic_scaling::reset_scale_A(
    cuMpSGEMM_handle *handle, const unsigned m, const unsigned n, T *const ptr,
    const unsigned ld, const unsigned stride, const unsigned batch_size,
    const unsigned exp_stats_buffer_id,
    const unsigned dynamic_launch_buffer_id) {
  reset_scale_AB(handle, m, n, ptr, ld, stride, batch_size, exp_stats_buffer_id,
                 dynamic_launch_buffer_id, matrix_A);
}
template void cumpsgemm::dynamic_scaling::reset_scale_A<float>(
    cuMpSGEMM_handle *, const unsigned, const unsigned, float *const,
    const unsigned, const unsigned, const unsigned, const unsigned,
    const unsigned);
template void cumpsgemm::dynamic_scaling::reset_scale_A<cuComplex>(
    cuMpSGEMM_handle *, const unsigned, const unsigned, cuComplex *const,
    const unsigned, const unsigned, const unsigned, const unsigned,
    const unsigned);
template <class T>
void cumpsgemm::dynamic_scaling::reset_scale_B(
    cuMpSGEMM_handle *handle, const unsigned m, const unsigned n, T *const ptr,
    const unsigned ld, const unsigned stride, const unsigned batch_size,
    const unsigned exp_stats_buffer_id,
    const unsigned dynamic_launch_buffer_id) {
  reset_scale_AB(handle, m, n, ptr, ld, stride, batch_size, exp_stats_buffer_id,
                 dynamic_launch_buffer_id, matrix_B);
}
template void cumpsgemm::dynamic_scaling::reset_scale_B<float>(
    cuMpSGEMM_handle *, const unsigned, const unsigned, float *const,
    const unsigned, const unsigned, const unsigned, const unsigned,
    const unsigned);
template void cumpsgemm::dynamic_scaling::reset_scale_B<cuComplex>(
    cuMpSGEMM_handle *, const unsigned, const unsigned, cuComplex *const,
    const unsigned, const unsigned, const unsigned, const unsigned,
    const unsigned);

float cumpsgemm::dynamic_scaling::get_max_exp(
    cuMpSGEMM_handle *handle, const unsigned exp_stats_buffer_id) {
  float max_exp;
  CUTF_CHECK_ERROR(cudaMemcpyAsync(
      &max_exp,
      handle->exp_stats_handle->dev_max_abs_buffer + exp_stats_buffer_id,
      sizeof(float), cudaMemcpyDefault, handle->cuda_stream));
  CUTF_CHECK_ERROR(cudaStreamSynchronize(handle->cuda_stream));
  return max_exp;
}

namespace {
__global__ void set_dynamic_launch_flag_by_exp_stats_kernel(
    int *const dynamic_mode_flag_buffer,
    const int *const A_exp_stats_compute_mode,
    const int *const B_exp_stats_compute_mode) {
  const auto pA = *A_exp_stats_compute_mode;
  const auto pB = *B_exp_stats_compute_mode;

  if (pA == CUMPSGEMM_TF32TCEC || pB == CUMPSGEMM_TF32TCEC) {
    *dynamic_mode_flag_buffer = CUMPSGEMM_TF32TCEC;
    return;
  }

  int flag = CUMPSGEMM_FP16TCEC;
  if (pA == CUMPSGEMM_FP16TCEC_SCALING) {
    cumpsgemm::dynamic_launch::utils::set_scale_A_flag(flag, true);
  }
  if (pB == CUMPSGEMM_FP16TCEC_SCALING) {
    cumpsgemm::dynamic_launch::utils::set_scale_B_flag(flag, true);
  }

  *dynamic_mode_flag_buffer = flag;
}
} // unnamed namespace

void cumpsgemm::dynamic_scaling::set_dynamic_launch_buffer_by_exp_stats(
    cuMpSGEMM_handle_t handle, const unsigned dynamic_mode_flag_id,
    const unsigned A_exp_stats_buffer_id,
    const unsigned B_exp_stats_buffer_id) {
  set_dynamic_launch_flag_by_exp_stats_kernel<<<1, 1, 0, handle->cuda_stream>>>(
      handle->dynamic_launch_handle->flag_buffer + dynamic_mode_flag_id,
      handle->exp_stats_handle->dev_compute_mode_buffer + A_exp_stats_buffer_id,
      handle->exp_stats_handle->dev_compute_mode_buffer +
          B_exp_stats_buffer_id);
}
