#include <cassert>
#include <cumpsgemm/cumpsgemm.hpp>
#include <cutf/cuda.hpp>
#include <cutf/memory.hpp>
#include <iostream>
#include <type_traits>

#include "device_common.hpp"
#include "dynamic_launch.hpp"
#include "dynamic_launch_utils.hpp"
#include "dynamic_scaling.hpp"
#include "exp_stats.hpp"
#include "handle.hpp"

// For debug
// #define CUMPSGEMM_CHECK_KERNEL_ERROR

namespace {
template <class T>
cumpsgemm::kernel_module_code::code_t
gen_module_code(const cublasOperation_t op_A, const cublasOperation_t op_B,
                const cuMpSGEMM_compute_mode_t compute_mode) {
  cumpsgemm::kernel_module_code::code_t code = 0;
  switch (compute_mode) {
  case CUMPSGEMM_FP16TC:
    code |= cumpsgemm::kernel_module_code::half |
            cumpsgemm::kernel_module_code::without_ec;
    break;
  case CUMPSGEMM_FP16TCEC:
    code |= cumpsgemm::kernel_module_code::half |
            cumpsgemm::kernel_module_code::with_ec;
    break;
  case CUMPSGEMM_TF32TC:
    code |= cumpsgemm::kernel_module_code::tf32 |
            cumpsgemm::kernel_module_code::without_ec;
    break;
  case CUMPSGEMM_TF32TCEC:
    code |= cumpsgemm::kernel_module_code::tf32 |
            cumpsgemm::kernel_module_code::with_ec;
    break;
  case CUMPSGEMM_FP32_SIMT:
    code |= cumpsgemm::kernel_module_code::simt |
            cumpsgemm::kernel_module_code::without_ec;
    break;
  default:
    break;
  }
  auto op_A_ = op_A;
  if (std::is_same<T, float>::value && op_A == CUBLAS_OP_C)
    op_A_ = CUBLAS_OP_T;
  switch (op_A_) {
  case CUBLAS_OP_N:
    code |= cumpsgemm::kernel_module_code::op_a_col_major;
    break;
  case CUBLAS_OP_T:
    code |= cumpsgemm::kernel_module_code::op_a_row_major;
    break;
  case CUBLAS_OP_C:
    code |= cumpsgemm::kernel_module_code::op_a_conjugate;
    break;
  default:
    break;
  }
  auto op_B_ = op_B;
  if (std::is_same<T, float>::value && op_B == CUBLAS_OP_C)
    op_B_ = CUBLAS_OP_T;
  switch (op_B_) {
  case CUBLAS_OP_N:
    code |= cumpsgemm::kernel_module_code::op_b_col_major;
    break;
  case CUBLAS_OP_T:
    code |= cumpsgemm::kernel_module_code::op_b_row_major;
    break;
  case CUBLAS_OP_C:
    code |= cumpsgemm::kernel_module_code::op_b_conjugate;
    break;
  default:
    break;
  }
  if (std::is_same<T, float>::value) {
    code |= cumpsgemm::kernel_module_code::s;
  } else if (std::is_same<T, cuComplex>::value) {
    code |= cumpsgemm::kernel_module_code::c;
  }

  assert(code <= cumpsgemm::kernel_module_code::max_code);

  return code;
}

template <class T>
void launch_kernel(const cumpsgemm::gemm_module gemm_module,
                   const int *const dynamic_launch_buffer_ptr,
                   const std::size_t m, const std::size_t n,
                   const std::size_t k, const T alpha, const T *const a_ptr,
                   const std::size_t lda, const T *const b_ptr,
                   const std::size_t ldb, const T beta, T *const c_ptr,
                   const std::size_t ldc, cudaStream_t cuda_stream) {
  const auto kernel_ptr = reinterpret_cast<cumpsgemm::gemm_kernel_func_t<T>>(
      gemm_module.kernel_func);
  const dim3 block_size(gemm_module.block_size);
  const dim3 grid_size(((m + gemm_module.smem_m - 1) / gemm_module.smem_m) *
                       ((n + gemm_module.smem_n - 1) / gemm_module.smem_n));

  kernel_ptr<<<grid_size, block_size, gemm_module.smem_size, cuda_stream>>>(
      dynamic_launch_buffer_ptr, m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta,
      c_ptr, ldc);
#ifdef CUMPSGEMM_CHECK_KERNEL_ERROR
  CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
#endif
}

template <class T>
void launch_atomic_kernel(const cumpsgemm::gemm_module gemm_module,
                          const int *const dynamic_launch_buffer_ptr,
                          const std::size_t m, const std::size_t n,
                          const std::size_t k, const T alpha,
                          const T *const a_ptr, const std::size_t lda,
                          const T *const b_ptr, const std::size_t ldb,
                          const T beta, T *const c_ptr, const std::size_t ldc,
                          cudaStream_t cuda_stream) {
  const auto kernel_ptr = reinterpret_cast<cumpsgemm::gemm_kernel_func_t<T>>(
      gemm_module.kernel_func);
  const dim3 block_size(gemm_module.block_size);
  const dim3 grid_size(((m + gemm_module.smem_m - 1) / gemm_module.smem_m) *
                       ((n + gemm_module.smem_n - 1) / gemm_module.smem_n) *
                       ((k + gemm_module.k_per_mn - 1) / gemm_module.k_per_mn));

  kernel_ptr<<<grid_size, block_size, gemm_module.smem_size, cuda_stream>>>(
      dynamic_launch_buffer_ptr, m, n, k, alpha, a_ptr, lda, b_ptr, ldb, beta,
      c_ptr, ldc);
#ifdef CUMPSGEMM_CHECK_KERNEL_ERROR
  CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
#endif
}

template <class T>
void launch_kernel(const cumpsgemm::gemm_module gemm_module,
                   const int *const dynamic_launch_buffer_ptr,
                   const std::size_t m, const std::size_t n,
                   const std::size_t k, const T alpha, const T *const a_ptr,
                   const std::size_t lda, const uint64_t stridea,
                   const T *const b_ptr, const std::size_t ldb,
                   const uint64_t strideb, const T beta, T *const c_ptr,
                   const std::size_t ldc, const uint64_t stridec,
                   const uint64_t batch_count, cudaStream_t cuda_stream) {
  const auto kernel_ptr =
      reinterpret_cast<cumpsgemm::gemm_stridedBatch_kernel_func_t<T>>(
          gemm_module.kernel_func);
  const dim3 block_size(gemm_module.block_size);
  const auto num_blocks_per_gemm =
      (m + gemm_module.smem_m - 1) / gemm_module.smem_m *
      (n + gemm_module.smem_n - 1) / gemm_module.smem_n;
  const dim3 grid_size(num_blocks_per_gemm * batch_count);

  kernel_ptr<<<grid_size, block_size, gemm_module.smem_size, cuda_stream>>>(
      dynamic_launch_buffer_ptr, m, n, k, alpha, a_ptr, lda, stridea, b_ptr,
      ldb, strideb, beta, c_ptr, ldc, stridec, num_blocks_per_gemm);
#ifdef CUMPSGEMM_CHECK_KERNEL_ERROR
  CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));
#endif
}

template <class T>
__global__ void fill_zero_kernel(T *const ptr, const unsigned m,
                                 const unsigned n, const std::uint64_t ld) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= m * n) {
    return;
  }
  const auto im = tid % m;
  const auto in = tid / m;
  const auto index = im + in * ld;

  ptr[index] = cumpsgemm::device::zero<T>();
}

template <class T>
void fill_zero(T *const ptr, const unsigned m, const unsigned n,
               const std::uint64_t ld, cudaStream_t cuda_stream) {
  const auto block_size = 256;
  const auto grid_size = (m * n + block_size - 1) / block_size;

  fill_zero_kernel<T><<<grid_size, block_size, 0, cuda_stream>>>(ptr, m, n, ld);
}

template <class T>
__global__ void post_atomic_kernel(T *const c_ptr, const T *const tmp_ptr,
                                   const unsigned m, const unsigned n,
                                   const std::uint64_t ldc,
                                   const std::uint64_t ldt, const T beta) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= m * n) {
    return;
  }
  const auto im = tid % m;
  const auto in = tid / m;
  const auto c_index = im + in * ldc;
  const auto t_index = im + in * ldt;

  c_ptr[c_index] =
      cumpsgemm::device::mad(c_ptr[c_index], beta, tmp_ptr[t_index]);
}

template <class T>
void post_atomic(T *const c_ptr, const T *const tmp_ptr, const unsigned m,
                 const unsigned n, const std::uint64_t ldc,
                 const std::uint64_t ldt, const T beta,
                 cudaStream_t cuda_stream) {
  const auto block_size = 256;
  const auto grid_size = (m * n + block_size - 1) / block_size;

  post_atomic_kernel<T><<<grid_size, block_size, 0, cuda_stream>>>(
      c_ptr, tmp_ptr, m, n, ldc, ldt, beta);
}
} // unnamed namespace

void init_temp_working_memory(cuMpSGEMM_handle *handle) {
  const auto float_count = (1lu << 22);
  handle->temp_working_memory_float_count = float_count;
  handle->temp_working_memory = cutf::memory::malloc<float>(float_count);
}

void destroy_temp_working_memory(cuMpSGEMM_handle *handle) {
  cutf::memory::free(handle->temp_working_memory);
}

template <class T>
cublasStatus_t
cumpsgemm::gemm(cuMpSGEMM_handle_t handle, const cublasOperation_t op_A,
                const cublasOperation_t op_B, const uint64_t m,
                const uint64_t n, const uint64_t k, const T *alpha,
                const T *const a_dmem_ptr, const uint64_t lda,
                const T *const b_dmem_ptr, const uint64_t ldb, const T *beta,
                T *const c_dmem_ptr, const uint64_t ldc,
                const cuMpSGEMM_compute_mode_t compute_mode,
                unsigned *const used_kernel_modeule_id) {

  if (compute_mode != CUMPSGEMM_AUTO) {
    const auto code = gen_module_code<T>(op_A, op_B, compute_mode);

    if (m * n >=
        (handle->temp_working_memory_float_count * sizeof(T) / sizeof(float))) {
      const auto kernel_module_candidate_list = handle->gemm_module[code];

      unsigned module_id;
      auto gemm_module =
          kernel_module_candidate_list[cumpsgemm::num_kernel_candidates - 1];
      for (module_id = 0; module_id < cumpsgemm::num_kernel_candidates - 1;
           module_id++) {
        const auto module = kernel_module_candidate_list[module_id];
        if (m * n / (module.smem_m * module.smem_n) >
            handle->num_sms * 2 /*A magic number :) */) {
          gemm_module = module;
          break;
        }
      }

      if (used_kernel_modeule_id != nullptr) {
        *used_kernel_modeule_id = module_id;
      }

      if (handle->exp_stats_handle->profiling_enabled) {
        handle->exp_stats_handle->profiler.start_timer_sync("gemm_kernel");
      }
      launch_kernel<T>(gemm_module, nullptr, m, n, k, *alpha, a_dmem_ptr, lda,
                       b_dmem_ptr, ldb, *beta, c_dmem_ptr, ldc,
                       handle->cuda_stream);
      if (handle->exp_stats_handle->profiling_enabled) {
        handle->exp_stats_handle->profiler.stop_timer_sync("gemm_kernel");
      }
    } else {
      T *r_c_dmem_ptr = c_dmem_ptr;
      uint64_t r_ldc = ldc;
      if (!cumpsgemm::device::is_zero(*beta)) {
        r_c_dmem_ptr = reinterpret_cast<T *>(handle->temp_working_memory);
        r_ldc = m;
      }
      // Initialize working memory
      fill_zero(r_c_dmem_ptr, m, n, r_ldc, handle->cuda_stream);

      // Main GEMM
      if (used_kernel_modeule_id != nullptr) {
        *used_kernel_modeule_id = 100;
      }
      const auto gemm_module = handle->gemm_atomic_module[code];

      if (handle->exp_stats_handle->profiling_enabled) {
        handle->exp_stats_handle->profiler.start_timer_sync("gemm_kernel");
      }
      launch_atomic_kernel<T>(gemm_module, nullptr, m, n, k, *alpha, a_dmem_ptr,
                              lda, b_dmem_ptr, ldb, *beta, r_c_dmem_ptr, r_ldc,
                              handle->cuda_stream);

      // post process if needed
      if (!cumpsgemm::device::is_zero(*beta)) {
        post_atomic(c_dmem_ptr,
                    reinterpret_cast<T *>(handle->temp_working_memory), m, n,
                    ldc, m, *beta, handle->cuda_stream);
      }
      if (handle->exp_stats_handle->profiling_enabled) {
        handle->exp_stats_handle->profiler.stop_timer_sync("gemm_kernel");
      }
    }
  } else {
    const auto code_A =
        gen_module_code<T>(op_A, op_B, handle->dynamic_launch_handle->mode_A);
    const auto code_B =
        gen_module_code<T>(op_A, op_B, handle->dynamic_launch_handle->mode_B);

    if (m * n >=
        (handle->temp_working_memory_float_count * sizeof(T) / sizeof(float))) {
      const auto kernel_module_candidate_list_A = handle->gemm_module[code_A];
      const auto kernel_module_candidate_list_B = handle->gemm_module[code_B];

      unsigned module_id;
      auto gemm_module_A =
          kernel_module_candidate_list_A[cumpsgemm::num_kernel_candidates - 1];
      auto gemm_module_B =
          kernel_module_candidate_list_B[cumpsgemm::num_kernel_candidates - 1];

      for (module_id = 0; module_id < cumpsgemm::num_kernel_candidates - 1;
           module_id++) {
        const auto module = kernel_module_candidate_list_A[module_id];
        if (m * n / (module.smem_m * module.smem_n) >
            handle->num_sms * 2 /*A magic number :) */) {
          gemm_module_A = module;
          break;
        }
      }

      for (module_id = 0; module_id < cumpsgemm::num_kernel_candidates - 1;
           module_id++) {
        const auto module = kernel_module_candidate_list_B[module_id];
        if (m * n / (module.smem_m * module.smem_n) >
            handle->num_sms * 2 /*A magic number :) */) {
          gemm_module_B = module;
          break;
        }
      }

      if (used_kernel_modeule_id != nullptr) {
        *used_kernel_modeule_id = 100;
      }

      if (handle->exp_stats_handle->profiling_enabled) {
        handle->exp_stats_handle->profiler.start_timer_sync("gemm_kernel_A");
      }
      launch_kernel<T>(gemm_module_A,
                       handle->dynamic_launch_handle->flag_buffer +
                           handle->dynamic_launch_handle->enabled_id,
                       m, n, k, *alpha, a_dmem_ptr, lda, b_dmem_ptr, ldb, *beta,
                       c_dmem_ptr, ldc, handle->cuda_stream);
      if (handle->exp_stats_handle->profiling_enabled) {
        handle->exp_stats_handle->profiler.stop_timer_sync("gemm_kernel_A");
      }

      if (handle->exp_stats_handle->profiling_enabled) {
        handle->exp_stats_handle->profiler.start_timer_sync("gemm_kernel_B");
      }
      launch_kernel<T>(gemm_module_B,
                       handle->dynamic_launch_handle->flag_buffer +
                           handle->dynamic_launch_handle->enabled_id,
                       m, n, k, *alpha, a_dmem_ptr, lda, b_dmem_ptr, ldb, *beta,
                       c_dmem_ptr, ldc, handle->cuda_stream);
      if (handle->exp_stats_handle->profiling_enabled) {
        handle->exp_stats_handle->profiler.stop_timer_sync("gemm_kernel_B");
      }
    } else {
      T *r_c_dmem_ptr = c_dmem_ptr;
      uint64_t r_ldc = ldc;
      if (!cumpsgemm::device::is_zero(*beta)) {
        r_c_dmem_ptr = reinterpret_cast<T *>(handle->temp_working_memory);
        r_ldc = m;
      }
      // Initialize working memory
      fill_zero(r_c_dmem_ptr, m, n, r_ldc, handle->cuda_stream);

      // Main GEMM
      const auto gemm_module_A = handle->gemm_atomic_module[code_A];
      const auto gemm_module_B = handle->gemm_atomic_module[code_B];

      if (used_kernel_modeule_id != nullptr) {
        *used_kernel_modeule_id = ~0u;
      }

      if (handle->exp_stats_handle->profiling_enabled) {
        handle->exp_stats_handle->profiler.start_timer_sync("gemm_kernel_A");
      }
      launch_kernel<T>(gemm_module_A,
                       handle->dynamic_launch_handle->flag_buffer +
                           handle->dynamic_launch_handle->enabled_id,
                       m, n, k, *alpha, a_dmem_ptr, lda, b_dmem_ptr, ldb, *beta,
                       r_c_dmem_ptr, r_ldc, handle->cuda_stream);
      if (handle->exp_stats_handle->profiling_enabled) {
        handle->exp_stats_handle->profiler.stop_timer_sync("gemm_kernel_A");
      }
      if (handle->exp_stats_handle->profiling_enabled) {
        handle->exp_stats_handle->profiler.start_timer_sync("gemm_kernel_B");
      }
      launch_kernel<T>(gemm_module_B,
                       handle->dynamic_launch_handle->flag_buffer +
                           handle->dynamic_launch_handle->enabled_id,
                       m, n, k, *alpha, a_dmem_ptr, lda, b_dmem_ptr, ldb, *beta,
                       r_c_dmem_ptr, r_ldc, handle->cuda_stream);
      if (handle->exp_stats_handle->profiling_enabled) {
        handle->exp_stats_handle->profiler.stop_timer_sync("gemm_kernel_B");
      }
      // post process if needed
      if (!cumpsgemm::device::is_zero(*beta)) {
        post_atomic(c_dmem_ptr,
                    reinterpret_cast<T *>(handle->temp_working_memory), m, n,
                    ldc, m, *beta, handle->cuda_stream);
      }
    }
  }

  return CUBLAS_STATUS_SUCCESS;
}

template <class T>
cublasStatus_t cumpsgemm::gemm_stridedBatch(
    cuMpSGEMM_handle_t handle, const cublasOperation_t op_A,
    const cublasOperation_t op_B, const uint64_t m, const uint64_t n,
    const uint64_t k, const T *alpha, const T *const a_dmem_ptr,
    const uint64_t lda, const uint64_t stridea, const T *const b_dmem_ptr,
    const uint64_t ldb, const uint64_t strideb, const T *beta,
    T *const c_dmem_ptr, const uint64_t ldc, const uint64_t stridec,
    const uint64_t batch_count, const cuMpSGEMM_compute_mode_t compute_mode,
    unsigned *const used_kernel_modeule_id) {
  if (m * n > (1lu << 24)) {
    for (std::uint64_t i = 0; i < batch_count; i++) {
      cumpsgemm::gemm(handle, op_A, op_B, m, n, k, alpha,
                      a_dmem_ptr + i * stridea, lda, b_dmem_ptr + i * strideb,
                      ldb, beta, c_dmem_ptr + i * stridec, ldc, compute_mode,
                      used_kernel_modeule_id);
    }
    return CUBLAS_STATUS_SUCCESS;
  }

  if (compute_mode != CUMPSGEMM_AUTO) {
    const auto code = gen_module_code<T>(op_A, op_B, compute_mode);

    const auto kernel_module_candidate_list =
        handle->gemm_stridedBatch_module[code];

    unsigned module_id;
    auto gemm_module =
        kernel_module_candidate_list[cumpsgemm::num_kernel_candidates - 1];
    for (module_id = 0; module_id < cumpsgemm::num_kernel_candidates - 1;
         module_id++) {
      const auto module = kernel_module_candidate_list[module_id];
      if (m * n / (module.smem_m * module.smem_n) * batch_count >
          handle->num_sms * 32 /*A magic number :) */) {
        gemm_module = module;
        break;
      }
    }

    if (used_kernel_modeule_id != nullptr) {
      *used_kernel_modeule_id = module_id;
    }

    if (handle->exp_stats_handle->profiling_enabled) {
      handle->exp_stats_handle->profiler.start_timer_sync(
          "batched_gemm_kernel");
    }
    launch_kernel<T>(gemm_module, nullptr, m, n, k, *alpha, a_dmem_ptr, lda,
                     stridea, b_dmem_ptr, ldb, strideb, *beta, c_dmem_ptr, ldc,
                     stridec, batch_count, handle->cuda_stream);
    if (handle->exp_stats_handle->profiling_enabled) {
      handle->exp_stats_handle->profiler.stop_timer_sync("batched_gemm_kernel");
    }
  } else {
    const auto code_A =
        gen_module_code<T>(op_A, op_B, handle->dynamic_launch_handle->mode_A);
    const auto code_B =
        gen_module_code<T>(op_A, op_B, handle->dynamic_launch_handle->mode_B);

    const auto kernel_module_candidate_list_A =
        handle->gemm_stridedBatch_module[code_A];
    const auto kernel_module_candidate_list_B =
        handle->gemm_stridedBatch_module[code_B];

    unsigned module_id;
    auto gemm_module_A =
        kernel_module_candidate_list_A[cumpsgemm::num_kernel_candidates - 1];
    auto gemm_module_B =
        kernel_module_candidate_list_B[cumpsgemm::num_kernel_candidates - 1];

    for (module_id = 0; module_id < cumpsgemm::num_kernel_candidates - 1;
         module_id++) {
      const auto module = kernel_module_candidate_list_A[module_id];
      if (m * n / (module.smem_m * module.smem_n) * batch_count >
          handle->num_sms * 32 /*A magic number :) */) {
        gemm_module_A = module;
        break;
      }
    }

    for (module_id = 0; module_id < cumpsgemm::num_kernel_candidates - 1;
         module_id++) {
      const auto module = kernel_module_candidate_list_B[module_id];
      if (m * n / (module.smem_m * module.smem_n) * batch_count >
          handle->num_sms * 32 /*A magic number :) */) {
        gemm_module_B = module;
        break;
      }
    }

    if (used_kernel_modeule_id != nullptr) {
      *used_kernel_modeule_id = ~0u;
    }

    if (handle->exp_stats_handle->profiling_enabled) {
      handle->exp_stats_handle->profiler.start_timer_sync(
          "batched_gemm_kernel_A");
    }
    launch_kernel<T>(gemm_module_A,
                     handle->dynamic_launch_handle->flag_buffer +
                         handle->dynamic_launch_handle->enabled_id,
                     m, n, k, *alpha, a_dmem_ptr, lda, stridea, b_dmem_ptr, ldb,
                     strideb, *beta, c_dmem_ptr, ldc, stridec, batch_count,
                     handle->cuda_stream);
    if (handle->exp_stats_handle->profiling_enabled) {
      handle->exp_stats_handle->profiler.stop_timer_sync(
          "batched_gemm_kernel_A");
    }

    if (handle->exp_stats_handle->profiling_enabled) {
      handle->exp_stats_handle->profiler.start_timer_sync(
          "batched_gemm_kernel_B");
    }
    launch_kernel<T>(gemm_module_B,
                     handle->dynamic_launch_handle->flag_buffer +
                         handle->dynamic_launch_handle->enabled_id,
                     m, n, k, *alpha, a_dmem_ptr, lda, stridea, b_dmem_ptr, ldb,
                     strideb, *beta, c_dmem_ptr, ldc, stridec, batch_count,
                     handle->cuda_stream);
    if (handle->exp_stats_handle->profiling_enabled) {
      handle->exp_stats_handle->profiler.stop_timer_sync(
          "batched_gemm_kernel_B");
    }
  }

  return CUBLAS_STATUS_SUCCESS;
}

extern "C" {
cublasStatus_t
cuMpSGEMM_sgemm(cuMpSGEMM_handle_t handle, const cublasOperation_t op_A,
                const cublasOperation_t op_B, const uint64_t m,
                const uint64_t n, const uint64_t k, const float *alpha,
                const float *const a_dmem_ptr, const uint64_t lda,
                const float *const b_dmem_ptr, const uint64_t ldb,
                const float *beta, float *const c_dmem_ptr, const uint64_t ldc,
                const cuMpSGEMM_compute_mode_t compute_mode) {
  assert(op_A != CUBLAS_OP_C);
  assert(op_B != CUBLAS_OP_C);
  return cumpsgemm::gemm<float>(handle, op_A, op_B, m, n, k, alpha, a_dmem_ptr,
                                lda, b_dmem_ptr, ldb, beta, c_dmem_ptr, ldc,
                                compute_mode);
}

cublasStatus_t cuMpSGEMM_cgemm(
    cuMpSGEMM_handle_t handle, const cublasOperation_t op_A,
    const cublasOperation_t op_B, const uint64_t m, const uint64_t n,
    const uint64_t k, const cuComplex *alpha, const cuComplex *const a_dmem_ptr,
    const uint64_t lda, const cuComplex *const b_dmem_ptr, const uint64_t ldb,
    const cuComplex *beta, cuComplex *const c_dmem_ptr, const uint64_t ldc,
    const cuMpSGEMM_compute_mode_t compute_mode) {
  return cumpsgemm::gemm<cuComplex>(handle, op_A, op_B, m, n, k, alpha,
                                    a_dmem_ptr, lda, b_dmem_ptr, ldb, beta,
                                    c_dmem_ptr, ldc, compute_mode);
}

cublasStatus_t cuMpSGEMM_sgemm_strided_batch(
    cuMpSGEMM_handle_t handle, const cublasOperation_t op_A,
    const cublasOperation_t op_B, const uint64_t m, const uint64_t n,
    const uint64_t k, const float *alpha, const float *const a_dmem_ptr,
    const uint64_t lda, const uint64_t stridea, const float *const b_dmem_ptr,
    const uint64_t ldb, const uint64_t strideb, const float *beta,
    float *const c_dmem_ptr, const uint64_t ldc, const uint64_t stridec,
    const uint64_t batch_count, const cuMpSGEMM_compute_mode_t compute_mode) {
  assert(op_A != CUBLAS_OP_C);
  assert(op_B != CUBLAS_OP_C);
  return cumpsgemm::gemm_stridedBatch<float>(
      handle, op_A, op_B, m, n, k, alpha, a_dmem_ptr, lda, stridea, b_dmem_ptr,
      ldb, strideb, beta, c_dmem_ptr, ldc, stridec, batch_count, compute_mode);
}

cublasStatus_t cuMpSGEMM_cgemm_strided_batch(
    cuMpSGEMM_handle_t handle, const cublasOperation_t op_A,
    const cublasOperation_t op_B, const uint64_t m, const uint64_t n,
    const uint64_t k, const cuComplex *alpha, const cuComplex *const a_dmem_ptr,
    const uint64_t lda, const uint64_t stridea,
    const cuComplex *const b_dmem_ptr, const uint64_t ldb,
    const uint64_t strideb, const cuComplex *beta, cuComplex *const c_dmem_ptr,
    const uint64_t ldc, const uint64_t stridec, const uint64_t batch_count,
    const cuMpSGEMM_compute_mode_t compute_mode) {
  return cumpsgemm::gemm_stridedBatch<cuComplex>(
      handle, op_A, op_B, m, n, k, alpha, a_dmem_ptr, lda, stridea, b_dmem_ptr,
      ldb, strideb, beta, c_dmem_ptr, ldc, stridec, batch_count, compute_mode);
}
} // extern "C"

std::pair<std::size_t, std::size_t>
cumpsgemm::get_exp_stats(cuMpSGEMM_handle_t handle, const unsigned buffer_id) {
  return cumpsgemm::exp_stats::get_exp_stats(handle, buffer_id);
}

unsigned cumpsgemm::get_current_exp_stats_buffer_id(cuMpSGEMM_handle_t handle) {
  return cumpsgemm::exp_stats::get_current_exp_stats_buffer_id(handle);
}

void cumpsgemm::reset_exp_stats_buffer_id(cuMpSGEMM_handle_t handle) {
  cumpsgemm::exp_stats::reset_exp_stats_buffer_id(handle);
}

cuMpSGEMM_compute_mode_t
cumpsgemm::get_exp_stats_compute_mode_level(cuMpSGEMM_handle_t handle,
                                            const unsigned buffer_id) {
  return cumpsgemm::exp_stats::get_compute_mode_level(handle, buffer_id);
}

float cumpsgemm::get_max_exp(cuMpSGEMM_handle_t handle,
                             const unsigned buffer_id) {
  return cumpsgemm::dynamic_scaling::get_max_exp(handle, buffer_id);
}

template <class T>
unsigned cumpsgemm::exp_stats_ext(cuMpSGEMM_handle_t handle, const unsigned m,
                                  const unsigned n, const T *const ptr,
                                  const unsigned ld, const unsigned batch_size,
                                  const unsigned stride) {
  cumpsgemm::exp_stats::exp_stats_ext(handle, m, n, ptr, ld, batch_size,
                                      stride);
  return cumpsgemm::exp_stats::get_current_exp_stats_buffer_id(handle);
}
template unsigned
cumpsgemm::exp_stats_ext<float>(cuMpSGEMM_handle_t, const unsigned,
                                const unsigned, const float *const,
                                const unsigned, const unsigned, const unsigned);
template unsigned cumpsgemm::exp_stats_ext<cuComplex>(
    cuMpSGEMM_handle_t, const unsigned, const unsigned, const cuComplex *const,
    const unsigned, const unsigned, const unsigned);

template <class T>
unsigned cumpsgemm::exp_max_ext(cuMpSGEMM_handle_t handle, const unsigned m,
                                const unsigned n, const T *const ptr,
                                const unsigned ld, const unsigned batch_size,
                                const unsigned stride) {
  cumpsgemm::exp_stats::exp_max_ext(handle, m, n, ptr, ld, batch_size, stride);
  return cumpsgemm::exp_stats::get_current_exp_stats_buffer_id(handle);
}
template unsigned cumpsgemm::exp_max_ext<float>(cuMpSGEMM_handle_t,
                                                const unsigned, const unsigned,
                                                const float *const,
                                                const unsigned, const unsigned,
                                                const unsigned);
template unsigned cumpsgemm::exp_max_ext<cuComplex>(
    cuMpSGEMM_handle_t, const unsigned, const unsigned, const cuComplex *const,
    const unsigned, const unsigned, const unsigned);

template <class T>
void cumpsgemm::scale_A(cuMpSGEMM_handle_t handle,
                        const unsigned exp_stats_buffer_id,
                        const unsigned dynamic_launch_flag_buffer_id,
                        const unsigned m, const unsigned n, T *const ptr,
                        const unsigned ld, const unsigned batch_size,
                        const unsigned stride) {
  cumpsgemm::dynamic_scaling::scale_A(handle, m, n, ptr, ld, stride, batch_size,
                                      exp_stats_buffer_id,
                                      dynamic_launch_flag_buffer_id);
}
template <class T>
void cumpsgemm::scale_B(cuMpSGEMM_handle_t handle,
                        const unsigned exp_stats_buffer_id,
                        const unsigned dynamic_launch_flag_buffer_id,
                        const unsigned m, const unsigned n, T *const ptr,
                        const unsigned ld, const unsigned batch_size,
                        const unsigned stride) {
  cumpsgemm::dynamic_scaling::scale_B(handle, m, n, ptr, ld, stride, batch_size,
                                      exp_stats_buffer_id,
                                      dynamic_launch_flag_buffer_id);
}
template void cumpsgemm::scale_A<float>(cuMpSGEMM_handle_t, const unsigned,
                                        const unsigned, const unsigned,
                                        const unsigned, float *const,
                                        const unsigned, const unsigned,
                                        const unsigned);
template void cumpsgemm::scale_A<cuComplex>(cuMpSGEMM_handle_t, const unsigned,
                                            const unsigned, const unsigned,
                                            const unsigned, cuComplex *const,
                                            const unsigned, const unsigned,
                                            const unsigned);
template void cumpsgemm::scale_B<float>(cuMpSGEMM_handle_t, const unsigned,
                                        const unsigned, const unsigned,
                                        const unsigned, float *const,
                                        const unsigned, const unsigned,
                                        const unsigned);
template void cumpsgemm::scale_B<cuComplex>(cuMpSGEMM_handle_t, const unsigned,
                                            const unsigned, const unsigned,
                                            const unsigned, cuComplex *const,
                                            const unsigned, const unsigned,
                                            const unsigned);

template <class T>
void cumpsgemm::scale_C(cuMpSGEMM_handle_t handle,
                        const unsigned exp_stats_buffer_A_id,
                        const unsigned exp_stats_buffer_B_id,
                        const unsigned dynamic_launch_flag_buffer_id,
                        const unsigned m, const unsigned n, T *const ptr,
                        const unsigned ld, const unsigned batch_size,
                        const unsigned stride) {
  cumpsgemm::dynamic_scaling::scale_C(
      handle, m, n, ptr, ld, stride, batch_size, exp_stats_buffer_A_id,
      exp_stats_buffer_B_id, dynamic_launch_flag_buffer_id);
}
template void cumpsgemm::scale_C<float>(cuMpSGEMM_handle_t, const unsigned,
                                        const unsigned, const unsigned,
                                        const unsigned, const unsigned,
                                        float *const, const unsigned,
                                        const unsigned, const unsigned);
template void cumpsgemm::scale_C<cuComplex>(cuMpSGEMM_handle_t, const unsigned,
                                            const unsigned, const unsigned,
                                            const unsigned, const unsigned,
                                            cuComplex *const, const unsigned,
                                            const unsigned, const unsigned);

template <class T>
void cumpsgemm::reset_scale_A(cuMpSGEMM_handle_t handle,
                              const unsigned exp_stats_buffer_id,
                              const unsigned dynamic_launch_flag_buffer_id,
                              const unsigned m, const unsigned n, T *const ptr,
                              const unsigned ld, const unsigned batch_size,
                              const unsigned stride) {
  cumpsgemm::dynamic_scaling::reset_scale_A(handle, m, n, ptr, ld, stride,
                                            batch_size, exp_stats_buffer_id,
                                            dynamic_launch_flag_buffer_id);
}
template <class T>
void cumpsgemm::reset_scale_B(cuMpSGEMM_handle_t handle,
                              const unsigned exp_stats_buffer_id,
                              const unsigned dynamic_launch_flag_buffer_id,
                              const unsigned m, const unsigned n, T *const ptr,
                              const unsigned ld, const unsigned batch_size,
                              const unsigned stride) {
  cumpsgemm::dynamic_scaling::reset_scale_B(handle, m, n, ptr, ld, stride,
                                            batch_size, exp_stats_buffer_id,
                                            dynamic_launch_flag_buffer_id);
}
template void cumpsgemm::reset_scale_A<float>(cuMpSGEMM_handle_t,
                                              const unsigned, const unsigned,
                                              const unsigned, const unsigned,
                                              float *const, const unsigned,
                                              const unsigned, const unsigned);
template void cumpsgemm::reset_scale_A<cuComplex>(
    cuMpSGEMM_handle_t, const unsigned, const unsigned, const unsigned,
    const unsigned, cuComplex *const, const unsigned, const unsigned,
    const unsigned);
template void cumpsgemm::reset_scale_B<float>(cuMpSGEMM_handle_t,
                                              const unsigned, const unsigned,
                                              const unsigned, const unsigned,
                                              float *const, const unsigned,
                                              const unsigned, const unsigned);
template void cumpsgemm::reset_scale_B<cuComplex>(
    cuMpSGEMM_handle_t, const unsigned, const unsigned, const unsigned,
    const unsigned, cuComplex *const, const unsigned, const unsigned,
    const unsigned);

unsigned
cumpsgemm::get_current_dynamic_launch_buffer_id(cuMpSGEMM_handle_t handle) {
  return cumpsgemm::dynamic_launch::get_current_dynamic_launch_flag_buffer_id(
      handle);
}

unsigned
cumpsgemm::get_next_dynamic_launch_buffer_id(cuMpSGEMM_handle_t handle) {
  return cumpsgemm::dynamic_launch::get_next_dynamic_launch_flag_buffer_id(
      handle);
}

cuMpSGEMM_compute_mode_t
cumpsgemm::get_dynamic_launch_gemm_compute_mode(cuMpSGEMM_handle_t handle,
                                                const unsigned buffer_id) {
  const auto mode =
      cumpsgemm::dynamic_launch::get_dynamic_launch_buffer(handle, buffer_id);

  return (cuMpSGEMM_compute_mode_t)
      cumpsgemm::dynamic_launch::utils::get_gemm_flag(mode);
}

std::pair<int, int>
cumpsgemm::get_dynamic_launch_scaling_mode_AB(cuMpSGEMM_handle_t handle,
                                              const unsigned buffer_id) {
  const auto mode =
      cumpsgemm::dynamic_launch::get_dynamic_launch_buffer(handle, buffer_id);

  return std::pair<int, int>{
      cumpsgemm::dynamic_launch::utils::get_scale_A_flag(mode),
      cumpsgemm::dynamic_launch::utils::get_scale_B_flag(mode)};
}

void cumpsgemm::set_dynamic_launch_buffer_by_exp_stats(
    cuMpSGEMM_handle *handle, const unsigned dynamic_launch_buffer_id,
    const unsigned A_exp_stats_buffer_id,
    const unsigned B_exp_stats_buffer_id) {
  cumpsgemm::dynamic_scaling::set_dynamic_launch_buffer_by_exp_stats(
      handle, dynamic_launch_buffer_id, A_exp_stats_buffer_id,
      B_exp_stats_buffer_id);
}

void cumpsgemm::enable_exp_stats_profiling(cuMpSGEMM_handle *const handle) {
  handle->exp_stats_handle->profiling_enabled = true;
  handle->exp_stats_handle->profiler.set_cuda_stream(handle->cuda_stream);
}

void cumpsgemm::disable_exp_stats_profiling(cuMpSGEMM_handle *const handle) {
  handle->exp_stats_handle->profiling_enabled = false;
}

void cumpsgemm::reset_exp_stats_profiling(cuMpSGEMM_handle *const handle) {
  handle->exp_stats_handle->profiler.clear();
}

void cumpsgemm::print_exp_stats_profiling(cuMpSGEMM_handle *const handle,
                                          unsigned csv) {
  if (csv) {
    handle->exp_stats_handle->profiler.print_result_csv(stdout);
  } else {
    handle->exp_stats_handle->profiler.print_result(stdout);
  }
}

std::unordered_map<std::string, cumpsgemm::debug::stats_t>
cumpsgemm::debug::get_exp_stats_profiling_result(
    cuMpSGEMM_handle *const handle) {
  const auto cutf_stats =
      handle->exp_stats_handle->profiler.get_statistics_list();

  std::unordered_map<std::string, cumpsgemm::debug::stats_t> result;
  for (const auto s : cutf_stats) {
    result.insert(std::make_pair(
        s.name, cumpsgemm::debug::stats_t{.time_sum = s.sum * 1e-9, .n = s.n}));
  }

  return result;
}
