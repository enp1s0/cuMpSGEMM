#include "exp_stats.hpp"
#include <cumpsgemm/cumpsgemm.hpp>
#include <cutf/experimental/fp.hpp>
#include <cutf/math.hpp>
#include <cutf/memory.hpp>
#include <thread>

namespace {
constexpr unsigned warp_size = 32;
}

// Ring buffer id calculator.
// 0 and 1 is reserved
// loop[2, 3, ..., buffer_length-1]
std::uint32_t
cumpsgemm::exp_stats::get_next_exp_stats_buffer_id(cuMpSGEMM_handle *handle) {
  handle->exp_stats_handle->current_buffer_id++;
  const auto next = handle->exp_stats_handle->current_buffer_id;
  if (next < handle->exp_stats_handle->buffer_length) {
    return next;
  }
  handle->exp_stats_handle->current_buffer_id = 2;
  return 2;
}

std::uint32_t cumpsgemm::exp_stats::get_current_exp_stats_buffer_id(
    cuMpSGEMM_handle *handle) {
  return handle->exp_stats_handle->current_buffer_id;
}

namespace {
__device__ float abs_max_float(const float a) { return cutf::math::abs(a); }
__device__ float abs_max_float(const cuComplex a) {
  return cutf::math::max(cutf::math::abs(a.x), cutf::math::abs(a.y));
}

template <class T> __device__ T make_zero() { return 0; }
template <> __device__ cuComplex make_zero() { return make_float2(0, 0); }

__device__ void update_count(const float a, unsigned &local_total_count,
                             unsigned &local_underflow_count,
                             const float ignore_threshold,
                             const float underflow_threshold) {
  const auto abs_a = cutf::math::abs(a);

  if (abs_a >= ignore_threshold) {
    local_total_count++;
    if (abs_a < underflow_threshold) {
      local_underflow_count++;
    }
  }
}

__device__ void update_count(const cuComplex a, unsigned &local_total_count,
                             unsigned &local_underflow_count,
                             const float ignore_threshold,
                             const float underflow_threshold) {
  update_count(a.x, local_total_count, local_underflow_count, ignore_threshold,
               underflow_threshold);
  update_count(a.y, local_total_count, local_underflow_count, ignore_threshold,
               underflow_threshold);
}

__global__ void
init_exp_stats_buffer(float *const max_exp_buffer,
                      cumpsgemm::counter_t *const total_count_buffer,
                      cumpsgemm::counter_t *const underflow_count_buffer) {
  *max_exp_buffer = 0;
  *total_count_buffer = 0;
  *underflow_count_buffer = 0;
}

// Get the largest abs value
template <unsigned BLOCK_SIZE, unsigned VEC_LEN, class LOOP_T, class T>
__global__ void exp_stats_ext_stage_1_kernel(
    float *const max_exp_buffer, cumpsgemm::counter_t *const total_count_buffer,
    cumpsgemm::counter_t *const underflow_count_buffer,
    const float underflow_threshold, const float ignore_threshold,
    const unsigned m, const unsigned n, const T *const ptr, const unsigned ld,
    const unsigned batch_size, const unsigned stride) {
  const auto ib = blockIdx.y;
  const auto local_mat_ptr = ptr + ib * stride;

  float local_max_abs_value = 0;
  unsigned local_total_count = 0;
  unsigned local_underflow_count = 0;

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
    } else {
      for (uint32_t i = 0; i < VEC_LEN; i++) {
        const auto gid = lid + i;
        T v;
        if (gid < m * n) {
          const auto im = gid % m;
          const auto in = gid / m;

          const auto memory_index = im + ld * in;
          v = local_mat_ptr[memory_index];
        } else {
          v = make_zero<T>();
        }
        vec[i] = v;
      }
    }
    for (uint32_t i = 0; i < VEC_LEN; i++) {
      local_max_abs_value =
          cutf::math::max(local_max_abs_value, abs_max_float(vec[i]));
      update_count(vec[i], local_total_count, local_underflow_count,
                   ignore_threshold, underflow_threshold);
    }
  }

  for (std::uint32_t offset = warp_size >> 1; offset >= 1; offset >>= 1) {
    local_max_abs_value = cutf::math::max(
        __shfl_xor_sync(~0u, local_max_abs_value, offset), local_max_abs_value);
    local_total_count += __shfl_xor_sync(~0u, local_total_count, offset);
    local_underflow_count +=
        __shfl_xor_sync(~0u, local_underflow_count, offset);
  }

  __shared__ float smem_max_abs_value[BLOCK_SIZE];
  __shared__ float smem_total_count[BLOCK_SIZE];
  __shared__ float smem_underflow_count[BLOCK_SIZE];

  if ((threadIdx.x & 0x1f) == 0) {
    smem_max_abs_value[threadIdx.x >> 5] = local_max_abs_value;
    smem_total_count[threadIdx.x >> 5] = local_total_count;
    smem_underflow_count[threadIdx.x >> 5] = local_underflow_count;
  }
  __syncthreads();

  if (threadIdx.x >= BLOCK_SIZE / warp_size)
    return;

  local_max_abs_value = smem_max_abs_value[threadIdx.x];
  local_total_count = smem_total_count[threadIdx.x];
  local_underflow_count = smem_underflow_count[threadIdx.x];

  for (std::uint32_t offset = (BLOCK_SIZE / warp_size) >> 1; offset >= 1;
       offset >>= 1) {
    local_max_abs_value = cutf::math::max(
        __shfl_xor_sync(~0u, local_max_abs_value, offset), local_max_abs_value);
    local_total_count += __shfl_xor_sync(~0u, local_total_count, offset);
    local_underflow_count +=
        __shfl_xor_sync(~0u, local_underflow_count, offset);
  }

  if (threadIdx.x == 0) {
    const std::uint32_t max_abs =
        cutf::experimental::fp::reinterpret_as_uint(local_max_abs_value) &
        0x7f800000u;
    atomicMax(reinterpret_cast<std::uint32_t *>(max_exp_buffer), max_abs);
    atomicAdd(total_count_buffer, local_total_count);
    atomicAdd(underflow_count_buffer, local_underflow_count);
  }
}

__global__ void
update_dynamic_mode_1_kernel(int *const dynamic_mode,
                             const float *const max_exp_buffer,
                             cumpsgemm::counter_t *const total_count_buffer,
                             cumpsgemm::counter_t *const underflow_count_buffer,
                             const float underflow_tolerance_rate) {
  if ((*underflow_count_buffer >
       (*total_count_buffer) * underflow_tolerance_rate) ||
      ((*max_exp_buffer) > 0 && (*total_count_buffer) == 0)) {
    *dynamic_mode = CUMPSGEMM_UNDEFINED;
    *total_count_buffer = 0;
    *underflow_count_buffer = 0;
  } else {
    *dynamic_mode = CUMPSGEMM_FP16TCEC;
  }
}

// exp_stats for cuBLAS original functions
template <unsigned BLOCK_SIZE, unsigned VEC_LEN, class LOOP_T, class T>
__global__ void exp_stats_ext_stage_2_kernel(
    const int *const dynamic_mode,
    cumpsgemm::counter_t *const total_count_buffer,
    cumpsgemm::counter_t *const underflow_count_buffer,
    const float *const max_exp_buffer, const float underflow_threshold,
    const float ignore_threshold, const unsigned m, const unsigned n,
    const T *const ptr, const unsigned ld, const unsigned batch_size,
    const unsigned stride) {
  // Launch-and-exit
  if (dynamic_mode == nullptr || *dynamic_mode != CUMPSGEMM_UNDEFINED) {
    return;
  }
  unsigned local_total_count = 0;
  unsigned local_underflow_count = 0;
  const auto ib = blockIdx.y;
  const auto local_mat_ptr = ptr + ib * stride;
  const auto max_exp_value = *max_exp_buffer;
  const auto abs_ignore_threshold = ignore_threshold * max_exp_value;
  const auto abs_underflow_threshold = underflow_threshold * max_exp_value;

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
    } else {
      for (uint32_t i = 0; i < VEC_LEN; i++) {
        const auto gid = lid + i;
        T v;
        if (gid < m * n) {
          const auto im = gid % m;
          const auto in = gid / m;

          const auto memory_index = im + ld * in;
          v = local_mat_ptr[memory_index];
        } else {
          v = make_zero<T>();
        }
        vec[i] = v;
      }
    }

    for (unsigned i = 0; i < VEC_LEN; i++) {
      update_count(vec[i], local_total_count, local_underflow_count,
                   abs_ignore_threshold, abs_underflow_threshold);
    }
  }

  for (std::uint32_t offset = warp_size >> 1; offset >= 1; offset >>= 1) {
    local_total_count += __shfl_xor_sync(~0u, local_total_count, offset);
    local_underflow_count +=
        __shfl_xor_sync(~0u, local_underflow_count, offset);
  }

  __shared__ float smem_total_count[BLOCK_SIZE];
  __shared__ float smem_underflow_count[BLOCK_SIZE];

  if ((threadIdx.x & 0x1f) == 0) {
    smem_total_count[threadIdx.x >> 5] = local_total_count;
    smem_underflow_count[threadIdx.x >> 5] = local_underflow_count;
  }
  __syncthreads();

  if (threadIdx.x >= BLOCK_SIZE / warp_size)
    return;

  local_total_count = smem_total_count[threadIdx.x];
  local_underflow_count = smem_underflow_count[threadIdx.x];

  for (std::uint32_t offset = (BLOCK_SIZE / warp_size) >> 1; offset >= 1;
       offset >>= 1) {
    local_total_count += __shfl_xor_sync(~0u, local_total_count, offset);
    local_underflow_count +=
        __shfl_xor_sync(~0u, local_underflow_count, offset);
  }

  if (threadIdx.x == 0) {
    atomicAdd(total_count_buffer, local_total_count);
    atomicAdd(underflow_count_buffer, local_underflow_count);
  }
}

__global__ void
update_dynamic_mode_2_kernel(int *const dynamic_mode,
                             cumpsgemm::counter_t *const total_count_buffer,
                             cumpsgemm::counter_t *const underflow_count_buffer,
                             const float underflow_tolerance_rate) {
  // Launch-and-exit
  if (dynamic_mode == nullptr || *dynamic_mode != CUMPSGEMM_UNDEFINED) {
    return;
  }

  if (*underflow_count_buffer <=
      (*total_count_buffer) * underflow_tolerance_rate) {
    *dynamic_mode = CUMPSGEMM_FP16TCEC_SCALING;
  } else {
    *dynamic_mode = CUMPSGEMM_TF32TCEC;
  }
}

template <class T, unsigned BLOCK_SIZE, unsigned VEC_LEN, class LOOP_T>
void launch_compute_mode_set_kernel(cuMpSGEMM_handle *handle, const unsigned m,
                                    const unsigned n, const T *const ptr,
                                    const unsigned ld,
                                    const unsigned batch_size,
                                    const unsigned stride,
                                    const unsigned buffer_id) {
  const dim3 grid_size(
      std::min<std::uint64_t>(
          ((1lu * m * n + BLOCK_SIZE - 1) / BLOCK_SIZE + VEC_LEN - 1) / VEC_LEN,
          handle->num_sms * 4),
      (stride == 0) ? 1 : batch_size);

  // 0
  if (handle->exp_stats_handle->profiling_enabled) {
    handle->exp_stats_handle->profiler.start_timer_sync("exp_stats_init");
  }
  init_exp_stats_buffer<<<1, 1, 0, handle->cuda_stream>>>(
      handle->exp_stats_handle->dev_max_abs_buffer + buffer_id,
      handle->exp_stats_handle->dev_total_count_buffer + buffer_id,
      handle->exp_stats_handle->dev_underflow_count_buffer + buffer_id);
  if (handle->exp_stats_handle->profiling_enabled) {
    handle->exp_stats_handle->profiler.stop_timer_sync("exp_stats_init");
  }

  // 1
  if (handle->exp_stats_handle->profiling_enabled) {
    handle->exp_stats_handle->profiler.start_timer_sync("exp_stats_1");
  }
  exp_stats_ext_stage_1_kernel<BLOCK_SIZE, VEC_LEN, LOOP_T, T>
      <<<grid_size, BLOCK_SIZE, 0, handle->cuda_stream>>>(
          handle->exp_stats_handle->dev_max_abs_buffer + buffer_id,
          handle->exp_stats_handle->dev_total_count_buffer + buffer_id,
          handle->exp_stats_handle->dev_underflow_count_buffer + buffer_id,
          handle->exp_stats_handle->underflow_threshold,
          handle->exp_stats_handle->ignore_threshold, m, n, ptr, ld, batch_size,
          stride);
  if (handle->exp_stats_handle->profiling_enabled) {
    handle->exp_stats_handle->profiler.stop_timer_sync("exp_stats_1");
  }

  // 2
  if (handle->exp_stats_handle->profiling_enabled) {
    handle->exp_stats_handle->profiler.start_timer_sync("exp_stats_set_1");
  }
  update_dynamic_mode_1_kernel<<<1, 1, 0, handle->cuda_stream>>>(
      handle->exp_stats_handle->dev_compute_mode_buffer + buffer_id,
      handle->exp_stats_handle->dev_max_abs_buffer + buffer_id,
      handle->exp_stats_handle->dev_total_count_buffer + buffer_id,
      handle->exp_stats_handle->dev_underflow_count_buffer + buffer_id,
      handle->exp_stats_handle->underflow_tolerance_rate);
  if (handle->exp_stats_handle->profiling_enabled) {
    handle->exp_stats_handle->profiler.stop_timer_sync("exp_stats_set_1");
  }

  // 3
  if (handle->exp_stats_handle->profiling_enabled) {
    handle->exp_stats_handle->profiler.start_timer_sync("exp_stats_2");
  }
  exp_stats_ext_stage_2_kernel<BLOCK_SIZE, VEC_LEN, LOOP_T, T>
      <<<grid_size, BLOCK_SIZE, 0, handle->cuda_stream>>>(
          handle->exp_stats_handle->dev_compute_mode_buffer + buffer_id,
          handle->exp_stats_handle->dev_total_count_buffer + buffer_id,
          handle->exp_stats_handle->dev_underflow_count_buffer + buffer_id,
          handle->exp_stats_handle->dev_max_abs_buffer + buffer_id,
          handle->exp_stats_handle->underflow_threshold / 2,
          handle->exp_stats_handle->ignore_threshold / 2, m, n, ptr, ld,
          batch_size, stride);
  if (handle->exp_stats_handle->profiling_enabled) {
    handle->exp_stats_handle->profiler.stop_timer_sync("exp_stats_2");
  }

  // 4
  if (handle->exp_stats_handle->profiling_enabled) {
    handle->exp_stats_handle->profiler.start_timer_sync("exp_stats_set_2");
  }
  update_dynamic_mode_2_kernel<<<1, 1, 0, handle->cuda_stream>>>(
      handle->exp_stats_handle->dev_compute_mode_buffer + buffer_id,
      handle->exp_stats_handle->dev_total_count_buffer + buffer_id,
      handle->exp_stats_handle->dev_underflow_count_buffer + buffer_id,
      handle->exp_stats_handle->underflow_tolerance_rate);
  if (handle->exp_stats_handle->profiling_enabled) {
    handle->exp_stats_handle->profiler.stop_timer_sync("exp_stats_set_2");
  }
}

// For FP16TCEC_SCALING
__global__ void init_exp_max(float *const max_exp_buffer) {
  *max_exp_buffer = 0;
}

template <unsigned BLOCK_SIZE, unsigned VEC_LEN, class LOOP_T, class T>
__global__ void exp_max_kernel(float *const max_exp_buffer, const unsigned m,
                               const unsigned n, const T *const ptr,
                               const unsigned ld, const unsigned batch_size,
                               const unsigned stride) {
  const auto ib = blockIdx.y;
  const auto local_mat_ptr = ptr + ib * stride;

  float local_max_abs_value = 0;

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
    } else {
      for (uint32_t i = 0; i < VEC_LEN; i++) {
        const auto gid = lid + i;
        T v;
        if (gid < m * n) {
          const auto im = gid % m;
          const auto in = gid / m;

          const auto memory_index = im + ld * in;
          v = local_mat_ptr[memory_index];
        } else {
          v = make_zero<T>();
        }
        vec[i] = v;
      }
    }
    for (uint32_t i = 0; i < VEC_LEN; i++) {
      local_max_abs_value =
          cutf::math::max(local_max_abs_value, abs_max_float(vec[i]));
    }
  }

  for (std::uint32_t offset = warp_size >> 1; offset >= 1; offset >>= 1) {
    local_max_abs_value = cutf::math::max(
        __shfl_xor_sync(~0u, local_max_abs_value, offset), local_max_abs_value);
  }

  __shared__ float smem_max_abs_value[BLOCK_SIZE];

  if ((threadIdx.x & 0x1f) == 0) {
    smem_max_abs_value[threadIdx.x >> 5] = local_max_abs_value;
  }
  __syncthreads();

  if (threadIdx.x >= BLOCK_SIZE / warp_size)
    return;

  local_max_abs_value = smem_max_abs_value[threadIdx.x];

  for (std::uint32_t offset = (BLOCK_SIZE / warp_size) >> 1; offset >= 1;
       offset >>= 1) {
    local_max_abs_value = cutf::math::max(
        __shfl_xor_sync(~0u, local_max_abs_value, offset), local_max_abs_value);
  }

  if (threadIdx.x == 0) {
    const std::uint32_t max_abs =
        cutf::experimental::fp::reinterpret_as_uint(local_max_abs_value) &
        0x7f800000u;
    atomicMax(reinterpret_cast<std::uint32_t *>(max_exp_buffer), max_abs);
  }
}

template <class T, unsigned BLOCK_SIZE, unsigned VEC_LEN, class LOOP_T>
void launch_max_exp_kernel(cuMpSGEMM_handle *handle, const unsigned m,
                           const unsigned n, const T *const ptr,
                           const unsigned ld, const unsigned batch_size,
                           const unsigned stride, const unsigned buffer_id) {
  const dim3 grid_size(
      std::min<std::uint64_t>(
          ((1lu * m * n + BLOCK_SIZE - 1) / BLOCK_SIZE + VEC_LEN - 1) / VEC_LEN,
          handle->num_sms * 4),
      (stride == 0) ? 1 : batch_size);

  // 0
  if (handle->exp_stats_handle->profiling_enabled) {
    handle->exp_stats_handle->profiler.start_timer_sync("exp_max_init");
  }
  init_exp_max<<<1, 1, 0, handle->cuda_stream>>>(
      handle->exp_stats_handle->dev_max_abs_buffer + buffer_id);
  if (handle->exp_stats_handle->profiling_enabled) {
    handle->exp_stats_handle->profiler.stop_timer_sync("exp_max_init");
  }

  // 1
  if (handle->exp_stats_handle->profiling_enabled) {
    handle->exp_stats_handle->profiler.start_timer_sync("exp_max");
  }
  exp_max_kernel<BLOCK_SIZE, VEC_LEN, LOOP_T, T>
      <<<grid_size, BLOCK_SIZE, 0, handle->cuda_stream>>>(
          handle->exp_stats_handle->dev_max_abs_buffer + buffer_id, m, n, ptr,
          ld, batch_size, stride);
  if (handle->exp_stats_handle->profiling_enabled) {
    handle->exp_stats_handle->profiler.stop_timer_sync("exp_max");
  }
}

// For init
__global__ void configure_buffer_kernel(int *const compute_mode_buffer) {
  compute_mode_buffer[0] = CUMPSGEMM_TF32TCEC;
  compute_mode_buffer[1] = CUMPSGEMM_FP16TCEC;
}
} // unnamed namespace

template <class T>
void cumpsgemm::exp_stats::exp_stats_ext(cuMpSGEMM_handle *handle,
                                         const unsigned m, const unsigned n,
                                         const T *const ptr, const unsigned ld,
                                         const unsigned batch_size,
                                         const unsigned stride) {
  const auto buffer_id =
      cumpsgemm::exp_stats::get_next_exp_stats_buffer_id(handle);
  using launch_func_t = void (*)(
      cuMpSGEMM_handle *, const unsigned, const unsigned, const T *const,
      const unsigned, const unsigned, const unsigned, const unsigned);
  launch_func_t launch_func;
  if (static_cast<std::size_t>(m) * n < (1lu << 15)) {
    launch_func = launch_compute_mode_set_kernel<T, 64, 4, unsigned>;
  } else if (static_cast<std::size_t>(m) * n < (1lu << 22)) {
    launch_func = launch_compute_mode_set_kernel<T, 128, 4, unsigned>;
  } else if (static_cast<std::size_t>(m) * n < (1lu << 32)) {
    launch_func = launch_compute_mode_set_kernel<T, 1024, 4, unsigned>;
  } else {
    launch_func = launch_compute_mode_set_kernel<T, 1024, 4, std::size_t>;
  }

  launch_func(handle, m, n, ptr, ld, batch_size, stride, buffer_id);
}

template void cumpsgemm::exp_stats::exp_stats_ext<float>(
    cuMpSGEMM_handle *, const unsigned, const unsigned, const float *const,
    const unsigned, const unsigned, const unsigned);
template void cumpsgemm::exp_stats::exp_stats_ext<cuComplex>(
    cuMpSGEMM_handle *, const unsigned, const unsigned, const cuComplex *const,
    const unsigned, const unsigned, const unsigned);

template <class T>
void cumpsgemm::exp_stats::exp_max_ext(cuMpSGEMM_handle *handle,
                                       const unsigned m, const unsigned n,
                                       const T *const ptr, const unsigned ld,
                                       const unsigned batch_size,
                                       const unsigned stride) {
  const auto buffer_id =
      cumpsgemm::exp_stats::get_next_exp_stats_buffer_id(handle);
  using launch_func_t = void (*)(
      cuMpSGEMM_handle *, const unsigned, const unsigned, const T *const,
      const unsigned, const unsigned, const unsigned, const unsigned);
  launch_func_t launch_func;
  if (static_cast<std::size_t>(m) * n < (1lu << 15)) {
    launch_func = launch_max_exp_kernel<T, 64, 4, unsigned>;
  } else if (static_cast<std::size_t>(m) * n < (1lu << 22)) {
    launch_func = launch_max_exp_kernel<T, 128, 4, unsigned>;
  } else if (static_cast<std::size_t>(m) * n < (1lu << 32)) {
    launch_func = launch_max_exp_kernel<T, 1024, 4, unsigned>;
  } else {
    launch_func = launch_max_exp_kernel<T, 1024, 4, std::size_t>;
  }

  launch_func(handle, m, n, ptr, ld, batch_size, stride, buffer_id);
}

template void cumpsgemm::exp_stats::exp_max_ext<float>(
    cuMpSGEMM_handle *, const unsigned, const unsigned, const float *const,
    const unsigned, const unsigned, const unsigned);
template void cumpsgemm::exp_stats::exp_max_ext<cuComplex>(
    cuMpSGEMM_handle *, const unsigned, const unsigned, const cuComplex *const,
    const unsigned, const unsigned, const unsigned);

void cumpsgemm::exp_stats::reset_exp_stats_buffer_id(cuMpSGEMM_handle *handle) {
  handle->exp_stats_handle->current_buffer_id = 1;
}

void init_exp_stats_counter_buffer(cuMpSGEMM_handle *handle) {
  handle->exp_stats_handle = new cumpsgemm::exp_stats::exp_stats_handle;
  cumpsgemm::exp_stats::reset_exp_stats_buffer_id(handle);

  handle->exp_stats_handle->enabled = false;
  handle->exp_stats_handle->buffer_length = 10000;
  handle->exp_stats_handle->ignore_threshold = 0;
  handle->exp_stats_handle->underflow_threshold = 1.f / (1u << 15);
  handle->exp_stats_handle->underflow_tolerance_rate = 0;
  handle->exp_stats_handle->counter_init_disabled = false;
  CUTF_CHECK_ERROR(cudaMalloc(
      &(handle->exp_stats_handle->dev_total_count_buffer),
      sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
  CUTF_CHECK_ERROR(cudaMalloc(
      &(handle->exp_stats_handle->dev_underflow_count_buffer),
      sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
  CUTF_CHECK_ERROR(
      cudaMalloc(&(handle->exp_stats_handle->dev_max_abs_buffer),
                 sizeof(float) * handle->exp_stats_handle->buffer_length));
  CUTF_CHECK_ERROR(
      cudaMalloc(&(handle->exp_stats_handle->dev_compute_mode_buffer),
                 sizeof(int) * handle->exp_stats_handle->buffer_length));

  configure_buffer_kernel<<<1, 1, 0, handle->cuda_stream>>>(
      handle->exp_stats_handle->dev_compute_mode_buffer);
}

void destroy_exp_stats_counter_buffer(cuMpSGEMM_handle *handle) {
  CUTF_CHECK_ERROR(cudaFree(handle->exp_stats_handle->dev_total_count_buffer));
  CUTF_CHECK_ERROR(
      cudaFree(handle->exp_stats_handle->dev_underflow_count_buffer));
  CUTF_CHECK_ERROR(cudaFree(handle->exp_stats_handle->dev_max_abs_buffer));
  CUTF_CHECK_ERROR(cudaFree(handle->exp_stats_handle->dev_compute_mode_buffer));

  delete handle->exp_stats_handle;
}

namespace {
__global__ void
download_value_kernel(std::size_t *const dst_ptr,
                      const cumpsgemm::counter_t *const src_ptr) {
  *dst_ptr = *src_ptr;
}
} // namespace

std::pair<std::size_t, std::size_t>
cumpsgemm::exp_stats::get_exp_stats(cuMpSGEMM_handle_t handle,
                                    const unsigned exp_stats_buffer_id) {
  auto host = cutf::memory::get_host_unique_ptr<std::size_t>(2);
  download_value_kernel<<<1, 1, 0, handle->cuda_stream>>>(
      host.get() + 0,
      handle->exp_stats_handle->dev_total_count_buffer + exp_stats_buffer_id);
  download_value_kernel<<<1, 1, 0, handle->cuda_stream>>>(
      host.get() + 1, handle->exp_stats_handle->dev_underflow_count_buffer +
                          exp_stats_buffer_id);

  CUTF_CHECK_ERROR(cudaStreamSynchronize(handle->cuda_stream));

  return std::pair<std::size_t, std::size_t>{host.get()[1], host.get()[0]};
}

void cumpsgemm::set_exp_stats_params(cuMpSGEMM_handle_t handle,
                                     const float ignore_threshold,
                                     const float underflow_threshold,
                                     const float underflow_tolerance_rate) {
  handle->exp_stats_handle->ignore_threshold = ignore_threshold;
  handle->exp_stats_handle->underflow_threshold = underflow_threshold;
  handle->exp_stats_handle->underflow_tolerance_rate = underflow_tolerance_rate;
}

void cumpsgemm::enable_exp_stats(cuMpSGEMM_handle_t handle) {
  handle->exp_stats_handle->enabled = true;
}

void cumpsgemm::disable_exp_stats(cuMpSGEMM_handle_t handle) {
  handle->exp_stats_handle->enabled = false;
}

void cumpsgemm::exp_stats::resize_counter(cuMpSGEMM_handle_t handle,
                                          const std::size_t new_length) {
  CUTF_CHECK_ERROR(cudaFree(handle->exp_stats_handle->dev_total_count_buffer));
  CUTF_CHECK_ERROR(
      cudaFree(handle->exp_stats_handle->dev_underflow_count_buffer));
  CUTF_CHECK_ERROR(cudaFree(handle->exp_stats_handle->dev_max_abs_buffer));
  CUTF_CHECK_ERROR(cudaFree(handle->exp_stats_handle->dev_compute_mode_buffer));

  handle->exp_stats_handle->buffer_length = new_length;

  CUTF_CHECK_ERROR(cudaMalloc(
      &(handle->exp_stats_handle->dev_total_count_buffer),
      sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
  CUTF_CHECK_ERROR(cudaMalloc(
      &(handle->exp_stats_handle->dev_underflow_count_buffer),
      sizeof(cumpsgemm::counter_t) * handle->exp_stats_handle->buffer_length));
  CUTF_CHECK_ERROR(
      cudaMalloc(&(handle->exp_stats_handle->dev_max_abs_buffer),
                 sizeof(float) * handle->exp_stats_handle->buffer_length));
  CUTF_CHECK_ERROR(
      cudaMalloc(&(handle->exp_stats_handle->dev_compute_mode_buffer),
                 sizeof(int) * handle->exp_stats_handle->buffer_length));

  configure_buffer_kernel<<<1, 1, 0, handle->cuda_stream>>>(
      handle->exp_stats_handle->dev_compute_mode_buffer);
}

cuMpSGEMM_compute_mode_t
cumpsgemm::exp_stats::get_compute_mode_level(cuMpSGEMM_handle *handle,
                                             const unsigned buffer_id) {
  int mode = 0;
  cutf::memory::copy(
      &mode, handle->exp_stats_handle->dev_compute_mode_buffer + buffer_id, 1);

  return (cuMpSGEMM_compute_mode_t)mode;
}
