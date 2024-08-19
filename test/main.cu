#include <chrono>
#include <cumpsgemm/cumpsgemm.hpp>
#include <cutf/cublas.hpp>
#include <cutf/curand.hpp>
#include <cutf/debug/time_breakdown.hpp>
#include <cutf/memory.hpp>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

// #define ENABLE_AUTO_MODE_PROFILING

constexpr unsigned test_count = 32;

enum gemm_type { s, c };

enum implementation_type {
  CUBLAS = CUMPSGEMM_CUBLAS,
  TF32TCEC = CUMPSGEMM_TF32TCEC,
  TF32TC = CUMPSGEMM_TF32TC,
  FP16TCEC = CUMPSGEMM_FP16TCEC,
  FP16TC = CUMPSGEMM_FP16TC,
  FP16TCEC_SCALING = CUMPSGEMM_FP16TCEC_SCALING,
  FP32_SIMT = CUMPSGEMM_FP32_SIMT,
};

cuMpSGEMM_compute_mode_t get_compute_mode(const implementation_type imp) {
  switch (imp) {
  case FP16TCEC_SCALING:
    return CUMPSGEMM_FP16TCEC;
  default:
    return (cuMpSGEMM_compute_mode_t)imp;
  }
}

bool is_scaling_enabled(const implementation_type imp) {
  switch (imp) {
  case FP16TCEC_SCALING:
    return true;
  default:
    return false;
  }
}

std::string get_implementation_type_name_str(const implementation_type imp) {
  switch (imp) {
  case CUBLAS:
    return "CUBLAS";
  case FP16TCEC_SCALING:
    return "FP16TCEC_SCALING";
  case FP16TCEC:
    return "FP16TCEC";
  case FP16TC:
    return "FP16TC";
  case TF32TCEC:
    return "TF32TCEC";
  case TF32TC:
    return "TF32TC";
  case FP32_SIMT:
    return "FP32_SIMT";
  default:
    return "Unknown(" + std::to_string(imp) + ")";
  }
}

double error_threshold(const cuMpSGEMM_compute_mode_t compute_mode,
                       const std::size_t N) {
  if (compute_mode == CUMPSGEMM_FP16TC || compute_mode == CUMPSGEMM_TF32TC) {
    return 1. / (1 << 10) * std::sqrt(N);
  }
  return 1. / (1 << 23) * std::sqrt(N);
}

__device__ double mad(const float a, const float b, const double c) {
  return static_cast<double>(a) * static_cast<double>(b) + c;
}

__device__ double mad(const double a, const float b, const double c) {
  return static_cast<double>(a) * static_cast<double>(b) + c;
}

__device__ double2 mad(const float2 a, const float2 b, const double2 c) {
  const auto dbl_a = cuComplexFloatToDouble(a);
  const auto dbl_b = cuComplexFloatToDouble(b);
  return cuCadd(cuCmul(dbl_a, dbl_b), c);
}

__device__ double2 mad(const double2 a, const float2 b, const double2 c) {
  const auto dbl_a = a;
  const auto dbl_b = cuComplexFloatToDouble(b);
  return cuCadd(cuCmul(dbl_a, dbl_b), c);
}

template <class T> struct doubled_t {
  using type = double;
};
template <> struct doubled_t<cuComplex> {
  using type = cuDoubleComplex;
};

template <class T>
__device__ T load_with_op(const T *const ptr, cublasOperation_t op) {
  return *ptr;
}

template <>
__device__ cuComplex load_with_op<cuComplex>(const cuComplex *const ptr,
                                             cublasOperation_t op) {
  if (op == CUBLAS_OP_C) {
    const auto v = *ptr;
    return cuConjf(v);
  }
  return *ptr;
}

__device__ double diff2(const cuDoubleComplex ab, const cuComplex c) {
  const auto real_diff = ab.x - c.x;
  const auto imag_diff = ab.y - c.y;
  return real_diff * real_diff + imag_diff * imag_diff;
}
__device__ double diff2(const double ab, const float c) {
  const auto diff = ab - c;
  return diff * diff;
}
__device__ double norm2(const cuDoubleComplex a) {
  return a.x * a.x + a.y * a.y;
}
__device__ double norm2(const double a) { return a * a; }

template <class T> __host__ __device__ T one() { return 1; }
template <> __host__ __device__ cuComplex one() { return make_cuComplex(1, 0); }
template <class T> __host__ __device__ T zero() { return 0; }
template <> __host__ __device__ cuComplex zero() {
  return make_cuComplex(0, 0);
}
template <> __host__ __device__ cuDoubleComplex zero() {
  return make_cuDoubleComplex(0, 0);
}

template <class T> __host__ __device__ bool is_zero(const T a) {
  return a == 0;
}
template <> __host__ __device__ bool is_zero<cuComplex>(const cuComplex a) {
  return a.x == 0 && a.y == 0;
}
template <>
__host__ __device__ bool is_zero<cuDoubleComplex>(const cuDoubleComplex a) {
  return a.x == 0 && a.y == 0;
}

template <class T>
__global__ void calc_matmul_residual_kernel(
    double *const base_norm2_ptr, double *const diff_norm2_ptr,
    const cublasOperation_t op_A, const cublasOperation_t op_B,
    const unsigned m, const unsigned n, const unsigned k, const T alpha,
    const T *const a_ptr, const unsigned lda, const T *const b_ptr,
    const unsigned ldb, const T beta, const T *const c_ptr, const unsigned ldc,
    const T *const r_ptr, const unsigned ldr) {
  const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= m * n)
    return;

  const auto c_m = tid % m;
  const auto c_n = tid / m;

  auto c = zero<typename doubled_t<T>::type>();
  for (std::size_t ik = 0; ik < k; ik++) {
    std::size_t a_index = 0;
    if (op_A == CUBLAS_OP_N) {
      a_index = c_m + ik * lda;
    } else {
      a_index = ik + c_m * lda;
    }

    std::size_t b_index = 0;
    if (op_B == CUBLAS_OP_N) {
      b_index = ik + c_n * ldb;
    } else {
      b_index = c_n + ik * ldb;
    }

    c = mad(load_with_op(a_ptr + a_index, op_A),
            load_with_op(b_ptr + b_index, op_B), c);
    const auto aa = load_with_op(a_ptr + a_index, op_A);
    const auto bb = load_with_op(b_ptr + b_index, op_B);
  }

  if (is_zero(beta) || c_ptr == nullptr) {
    c = mad(c, alpha, zero<typename doubled_t<T>::type>());
  } else {
    c = mad(
        c, alpha,
        mad(beta, c_ptr[c_m + c_n * ldc], zero<typename doubled_t<T>::type>()));
  }
  const auto base_norm2 = norm2(c);
  const auto diff_norm2 = diff2(c, r_ptr[c_m + c_n * ldc]);

  atomicAdd(base_norm2_ptr, base_norm2);
  atomicAdd(diff_norm2_ptr, diff_norm2);
}

template <class T>
double
calc_matmul_residual(const cublasOperation_t op_A, const cublasOperation_t op_B,
                     const unsigned m, const unsigned n, const unsigned k,
                     const T alpha, const T *const a_ptr, const unsigned lda,
                     const T *const b_ptr, const unsigned ldb, const T beta,
                     const T *const c_ptr, const unsigned ldc,
                     const T *const r_ptr, const unsigned ldr) {
  auto base_norm2_ptr = cutf::memory::malloc_managed<double>(1);
  auto diff_norm2_ptr = cutf::memory::malloc_managed<double>(1);

  *base_norm2_ptr = 0;
  *diff_norm2_ptr = 0;

  constexpr unsigned block_size = 256;
  const auto num_threads = m * n;
  const auto grid_size = (num_threads + block_size - 1) / block_size;

  cudaDeviceSynchronize();
  calc_matmul_residual_kernel<<<grid_size, block_size>>>(
      base_norm2_ptr, diff_norm2_ptr, op_A, op_B, m, n, k, alpha, a_ptr, lda,
      b_ptr, ldb, beta, c_ptr, ldc, r_ptr, ldr);
  cudaDeviceSynchronize();

  const auto residual = std::sqrt(*diff_norm2_ptr / *base_norm2_ptr);

  cutf::memory::free(base_norm2_ptr);
  cutf::memory::free(diff_norm2_ptr);

  return residual;
}

template <class T>
__global__ void copy_matrix_kernel(T *dst_ptr, const std::size_t ldd,
                                   const T *src_ptr, const std::size_t lds,
                                   const std::size_t m, const std::size_t n) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= m * n) {
    return;
  }

  const auto mi = tid % m;
  const auto ni = tid / m;

  dst_ptr[mi + mi * ldd] = src_ptr[mi + ni * lds];
}

template <class T>
void copy_matrix(T *dst_ptr, const std::size_t ldd, const T *src_ptr,
                 const std::size_t lds, const std::size_t m,
                 const std::size_t n) {
  constexpr std::size_t block_size = 256;
  copy_matrix_kernel<<<(m * n + block_size - 1) / block_size, block_size>>>(
      dst_ptr, ldd, src_ptr, lds, m, n);
}

void cublas_gemm(cublasHandle_t const cublas_handle,
                 const cublasOperation_t op_A, const cublasOperation_t op_B,
                 const unsigned m, const unsigned n, const unsigned k,
                 const float *const alpha, const float *const a_ptr,
                 const unsigned lda, const float *const b_ptr,
                 const unsigned ldb, const float *const beta,
                 float *const c_ptr, const unsigned ldc) {
  CUTF_CHECK_ERROR(cublasSgemm(cublas_handle, op_A, op_B, m, n, k, alpha, a_ptr,
                               lda, b_ptr, ldb, beta, c_ptr, ldc));
}

void cublas_gemm(cublasHandle_t const cublas_handle,
                 const cublasOperation_t op_A, const cublasOperation_t op_B,
                 const unsigned m, const unsigned n, const unsigned k,
                 const cuComplex *const alpha, const cuComplex *const a_ptr,
                 const unsigned lda, const cuComplex *const b_ptr,
                 const unsigned ldb, const cuComplex *const beta,
                 cuComplex *const c_ptr, const unsigned ldc) {
  CUTF_CHECK_ERROR(cublasCgemm(cublas_handle, op_A, op_B, m, n, k, alpha, a_ptr,
                               lda, b_ptr, ldb, beta, c_ptr, ldc));
}

void cublas_gemm_strided_batch(
    cublasHandle_t const cublas_handle, const cublasOperation_t op_A,
    const cublasOperation_t op_B, const unsigned m, const unsigned n,
    const unsigned k, const float *const alpha, const float *const a_ptr,
    const unsigned lda, const long long int stride_a, const float *const b_ptr,
    const unsigned ldb, const long long int stride_b, const float *const beta,
    float *const c_ptr, const unsigned ldc, const long long int stride_c,
    const long long int batch_count) {
  CUTF_CHECK_ERROR(cublasSgemmStridedBatched(
      cublas_handle, op_A, op_B, m, n, k, alpha, a_ptr, lda, stride_a, b_ptr,
      ldb, stride_b, beta, c_ptr, ldc, stride_c, batch_count));
}

void cublas_gemm_strided_batch(
    cublasHandle_t const cublas_handle, const cublasOperation_t op_A,
    const cublasOperation_t op_B, const unsigned m, const unsigned n,
    const unsigned k, const cuComplex *const alpha,
    const cuComplex *const a_ptr, const unsigned lda,
    const long long int stride_a, const cuComplex *const b_ptr,
    const unsigned ldb, const long long int stride_b,
    const cuComplex *const beta, cuComplex *const c_ptr, const unsigned ldc,
    const long long int stride_c, const long long int batch_count) {
  CUTF_CHECK_ERROR(cublasCgemmStridedBatched(
      cublas_handle, op_A, op_B, m, n, k, alpha, a_ptr, lda, stride_a, b_ptr,
      ldb, stride_b, beta, c_ptr, ldc, stride_c, batch_count));
}

template <class T>
int sgemm_test_core(cublasHandle_t const cublas_handle,
                    cuMpSGEMM_handle_t const cuMpSGEMM_handle,
                    const cublasOperation_t op_A, const cublasOperation_t op_B,
                    const unsigned m, const unsigned n, const unsigned k,
                    T *const a_ptr, const unsigned lda, T *const b_ptr,
                    const unsigned ldb, T *const c_ptr, const unsigned ldc,
                    T *const r_ptr, const unsigned ldr,
                    const cuMpSGEMM_compute_mode_t compute_mode,
                    const bool scaling = false) {
  const auto alpha = one<T>(), beta = zero<T>();

  unsigned module_stage = 0;
  unsigned exp_stats_id_A, exp_stats_id_B;
  auto gemm_func = [&](const bool reset_scaling = false) {
    if (compute_mode == CUMPSGEMM_CUBLAS) {
      cublas_gemm(cublas_handle, op_A, op_B, m, n, k, &alpha, a_ptr, lda, b_ptr,
                  ldb, &beta, c_ptr, ldc);
    } else {
      if (scaling) {
        cumpsgemm::exp_stats_ext(cuMpSGEMM_handle,
                                 (op_A == CUBLAS_OP_N ? m : k),
                                 (op_A == CUBLAS_OP_N ? k : m), a_ptr, lda);
        exp_stats_id_A =
            cumpsgemm::get_current_exp_stats_buffer_id(cuMpSGEMM_handle);
        cumpsgemm::scale_A(cuMpSGEMM_handle, exp_stats_id_A, 1,
                           (op_A == CUBLAS_OP_N ? m : k),
                           (op_A == CUBLAS_OP_N ? k : m), a_ptr, lda);

        cumpsgemm::exp_stats_ext(cuMpSGEMM_handle,
                                 (op_B == CUBLAS_OP_N ? k : n),
                                 (op_B == CUBLAS_OP_N ? n : k), b_ptr, ldb);
        exp_stats_id_B =
            cumpsgemm::get_current_exp_stats_buffer_id(cuMpSGEMM_handle);
        cumpsgemm::scale_B(cuMpSGEMM_handle, exp_stats_id_B, 1,
                           (op_B == CUBLAS_OP_N ? k : n),
                           (op_B == CUBLAS_OP_N ? n : k), b_ptr, ldb);
      }
      cumpsgemm::gemm(cuMpSGEMM_handle, op_A, op_B, m, n, k, &alpha, a_ptr, lda,
                      b_ptr, ldb, &beta, c_ptr, ldc, compute_mode,
                      &module_stage);
      if (scaling) {
        cumpsgemm::scale_C(cuMpSGEMM_handle, exp_stats_id_A, exp_stats_id_B, 1,
                           m, n, c_ptr, ldc);
        if (reset_scaling) {
          cumpsgemm::reset_scale_A(cuMpSGEMM_handle, exp_stats_id_A, 1,
                                   (op_A == CUBLAS_OP_N ? m : k),
                                   (op_A == CUBLAS_OP_N ? k : m), a_ptr, lda);
          cumpsgemm::reset_scale_B(cuMpSGEMM_handle, exp_stats_id_B, 1,
                                   (op_B == CUBLAS_OP_N ? k : n),
                                   (op_B == CUBLAS_OP_N ? n : k), b_ptr, ldb);
        }
      }
    }
  };

  // C to R
  copy_matrix(r_ptr, ldr, c_ptr, ldc, m, n);

  gemm_func(true);

  CUTF_CHECK_ERROR(cudaDeviceSynchronize());

  const auto residual =
      calc_matmul_residual(op_A, op_B, m, n, k, alpha, a_ptr, lda, b_ptr, ldb,
                           beta, r_ptr, ldr, c_ptr, ldc);
  const auto check = residual < error_threshold(compute_mode, k);

#ifdef ENABLE_AUTO_MODE_PROFILING
  cumpsgemm::enable_exp_stats_profiling(cuMpSGEMM_handle);
  cumpsgemm::reset_exp_stats_profiling(cuMpSGEMM_handle);
  cumpsgemm::set_exp_stats_params(cuMpSGEMM_handle, 1e-30, 1, 0);
#endif

  // Throughput
  CUTF_CHECK_ERROR(cudaDeviceSynchronize());
  const auto start_clock = std::chrono::system_clock::now();
  for (unsigned i = 0; i < test_count; i++) {
    gemm_func();
  }
  CUTF_CHECK_ERROR(cudaDeviceSynchronize());
  const auto end_clock = std::chrono::system_clock::now();
  const auto elapsed_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end_clock -
                                                            start_clock)
          .count() *
      1e-6;
  const auto throughput = 2lu * m * n * k *
                          (std::is_same<float, T>::value ? 1 : 4) /
                          (elapsed_time / test_count);

#ifdef ENABLE_AUTO_MODE_PROFILING
  cumpsgemm::print_exp_stats_profiling(cuMpSGEMM_handle);
#endif

  std::printf("%s,%s,%s,%s,%u,%u,%u,%e,%e,%s,%u\n",
              (std::is_same<float, T>::value ? "sgemm" : "cgemm"),
              cuMpSGEMM_get_compute_mode_string(compute_mode),
              (op_A == CUBLAS_OP_N) ? "N" : ((op_A == CUBLAS_OP_T) ? "T" : "C"),
              (op_B == CUBLAS_OP_N) ? "N" : ((op_B == CUBLAS_OP_T) ? "T" : "C"),
              m, n, k, throughput * 1e-12, residual, (check ? "OK" : "NG"),
              module_stage);
  std::fflush(stdout);

  if (check) {
    return 0;
  } else {
    return 1;
  }
}

template <class T>
int sgemm_strided_batch_test_core(
    cublasHandle_t const cublas_handle,
    cuMpSGEMM_handle_t const cuMpSGEMM_handle, const cublasOperation_t op_A,
    const cublasOperation_t op_B, const unsigned m, const unsigned n,
    const unsigned k, T *const a_ptr, const unsigned lda,
    const long long int stride_a, T *const b_ptr, const unsigned ldb,
    const long long int stride_b, T *const c_ptr, const unsigned ldc,
    const long long int stride_c, const long long int batch_count,
    const cuMpSGEMM_compute_mode_t compute_mode, const bool scaling = false) {
  const auto alpha = one<T>(), beta = zero<T>();

  unsigned module_stage = 0;

  auto gemm_func = [&](const bool reset_scaling = false) {
    if (compute_mode == CUMPSGEMM_CUBLAS) {
      cublas_gemm_strided_batch(cublas_handle, op_A, op_B, m, n, k, &alpha,
                                a_ptr, lda, stride_a, b_ptr, ldb, stride_b,
                                &beta, c_ptr, ldc, stride_c, batch_count);
    } else {
      unsigned exp_stats_id_A, exp_stats_id_B;
      if (scaling) {
        cumpsgemm::exp_stats_ext(
            cuMpSGEMM_handle, (op_A == CUBLAS_OP_N ? m : k),
            (op_A == CUBLAS_OP_N ? k : m), a_ptr, lda, batch_count, stride_a);
        exp_stats_id_A =
            cumpsgemm::get_current_exp_stats_buffer_id(cuMpSGEMM_handle);
        cumpsgemm::scale_A(
            cuMpSGEMM_handle, exp_stats_id_A, 1, (op_A == CUBLAS_OP_N ? m : k),
            (op_A == CUBLAS_OP_N ? k : m), a_ptr, lda, batch_count, stride_a);

        cumpsgemm::exp_stats_ext(
            cuMpSGEMM_handle, (op_B == CUBLAS_OP_N ? k : n),
            (op_B == CUBLAS_OP_N ? n : k), b_ptr, ldb, batch_count, stride_b);
        exp_stats_id_B =
            cumpsgemm::get_current_exp_stats_buffer_id(cuMpSGEMM_handle);
        cumpsgemm::scale_B(
            cuMpSGEMM_handle, exp_stats_id_B, 1, (op_B == CUBLAS_OP_N ? k : n),
            (op_B == CUBLAS_OP_N ? n : k), b_ptr, ldb, batch_count, stride_b);
      }
      cumpsgemm::gemm_stridedBatch(cuMpSGEMM_handle, op_A, op_B, m, n, k,
                                   &alpha, a_ptr, lda, stride_a, b_ptr, ldb,
                                   stride_b, &beta, c_ptr, ldc, stride_c,
                                   batch_count, compute_mode, &module_stage);
      if (scaling) {
        cumpsgemm::scale_C(cuMpSGEMM_handle, exp_stats_id_A, exp_stats_id_B, 1,
                           m, n, c_ptr, ldc, batch_count, stride_c);
        if (reset_scaling) {
          cumpsgemm::reset_scale_A(cuMpSGEMM_handle, exp_stats_id_A, 1,
                                   (op_A == CUBLAS_OP_N ? m : k),
                                   (op_A == CUBLAS_OP_N ? k : m), a_ptr, lda,
                                   batch_count, stride_a);
          cumpsgemm::reset_scale_B(cuMpSGEMM_handle, exp_stats_id_B, 1,
                                   (op_B == CUBLAS_OP_N ? k : n),
                                   (op_B == CUBLAS_OP_N ? n : k), b_ptr, ldb,
                                   batch_count, stride_b);
        }
      }
    }
  };

  gemm_func(true);

  CUTF_CHECK_ERROR(cudaDeviceSynchronize());

  double residual = 0;
  for (unsigned long long int b = 0; b < batch_count; b++) {
    residual += calc_matmul_residual(
        op_A, op_B, m, n, k, one<T>(), a_ptr + stride_a * b, lda,
        b_ptr + stride_b * b, ldb, zero<T>(), reinterpret_cast<T *>(0), 0,
        c_ptr + stride_c * b, ldc);
  }
  residual /= batch_count;
  const auto check = residual < error_threshold(compute_mode, m);

#ifdef ENABLE_AUTO_MODE_PROFILING
  cumpsgemm::enable_exp_stats_profiling(cuMpSGEMM_handle);
  cumpsgemm::reset_exp_stats_profiling(cuMpSGEMM_handle);
  cumpsgemm::set_exp_stats_params(cuMpSGEMM_handle, 1e-30, 1, 0);
#endif

  // Throughput
  CUTF_CHECK_ERROR(cudaDeviceSynchronize());
  const auto start_clock = std::chrono::system_clock::now();
  for (unsigned i = 0; i < test_count; i++) {
    gemm_func();
  }
  CUTF_CHECK_ERROR(cudaDeviceSynchronize());

  const auto end_clock = std::chrono::system_clock::now();
  const auto elapsed_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end_clock -
                                                            start_clock)
          .count() *
      1e-6;
  const auto throughput = 2lu * m * n * k * batch_count *
                          (std::is_same<float, T>::value ? 1 : 4) /
                          (elapsed_time / test_count);

#ifdef ENABLE_AUTO_MODE_PROFILING
  cumpsgemm::print_exp_stats_profiling(cuMpSGEMM_handle);
#endif

  std::printf("%s,%s,%s,%s,%u,%u,%u,%lld,%e,%e,%s,%u\n",
              (std::is_same<float, T>::value ? "sgemm" : "cgemm"),
              cuMpSGEMM_get_compute_mode_string(compute_mode),
              (op_A == CUBLAS_OP_N) ? "N" : ((op_A == CUBLAS_OP_T) ? "T" : "C"),
              (op_B == CUBLAS_OP_N) ? "N" : ((op_B == CUBLAS_OP_T) ? "T" : "C"),
              m, n, k, batch_count, throughput * 1e-12, residual,
              (check ? "OK" : "NG"), module_stage);
  std::fflush(stdout);

  if (check) {
    return 0;
  } else {
    return 1;
  }
}

void gemm_test(const std::size_t min_N, const std::size_t max_N,
               const std::size_t interval,
               const std::vector<implementation_type> &imp_list,
               const gemm_type gemm, const bool is_seq) {
  constexpr uint64_t seed = 0;
  const std::size_t max_num_elements =
      (is_seq ? max_N * max_N : (1lu << (2 * max_N))) *
      (gemm == gemm_type::c ? 2 : 1);
  float *a_ptr = cutf::memory::malloc<float>(max_num_elements);
  float *b_ptr = cutf::memory::malloc<float>(max_num_elements);
  float *c_ptr = cutf::memory::malloc<float>(max_num_elements);
  float *r_ptr = cutf::memory::malloc<float>(max_num_elements);

  auto curand_gen =
      cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_PHILOX4_32_10);
  CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*curand_gen.get(), seed));
  CUTF_CHECK_ERROR(cutf::curand::generate_normal(*curand_gen.get(), a_ptr,
                                                 max_num_elements, 0, 1));
  CUTF_CHECK_ERROR(cutf::curand::generate_normal(*curand_gen.get(), b_ptr,
                                                 max_num_elements, 0, 1));

  std::vector<cublasOperation_t> sgemm_ops = {CUBLAS_OP_N, CUBLAS_OP_T};
  std::vector<cublasOperation_t> cgemm_ops = {CUBLAS_OP_N, CUBLAS_OP_T,
                                              CUBLAS_OP_C};

  std::printf("## %s\n", __func__);
  std::printf("type,mode,op_A,op_B,m,n,k,throughput_in_tflops,residual,check,"
              "module_stage\n");
  unsigned num_tests = 0;
  unsigned num_passed = 0;
  auto cublas_handle_uptr = cutf::cublas::get_cublas_unique_ptr();
  cumpsgemm::handle_t cuMpSGEMM_handle;
  cumpsgemm::create(cuMpSGEMM_handle);

  std::vector<std::size_t> N_list;
  if (is_seq) {
    for (unsigned N = min_N; N <= max_N; N += interval) {
      N_list.push_back(N);
    }
  } else {
    for (unsigned N = min_N; N <= max_N; N += interval) {
      N_list.push_back(1lu << N);
    }
  }

  for (const auto imp : imp_list) {
    const auto mode = get_compute_mode(imp);
    const auto scaling = is_scaling_enabled(imp);
    if (gemm == gemm_type::s) {
      for (const auto op_A : sgemm_ops) {
        for (const auto op_B : sgemm_ops) {
          for (const auto N : N_list) {
            const auto res = sgemm_test_core(
                *cublas_handle_uptr.get(), cuMpSGEMM_handle, op_A, op_B, N, N,
                N, a_ptr, N, b_ptr, N, c_ptr, N, r_ptr, N, mode, scaling);
            num_tests++;
            if (res == 0) {
              num_passed++;
            }
          }
        }
      }
    } else if (gemm == gemm_type::c) {
      for (const auto op_A : cgemm_ops) {
        for (const auto op_B : cgemm_ops) {
          for (const auto N : N_list) {
            const auto res = sgemm_test_core(
                *cublas_handle_uptr.get(), cuMpSGEMM_handle, op_A, op_B, N, N,
                N, reinterpret_cast<cuComplex *>(a_ptr), N,
                reinterpret_cast<cuComplex *>(b_ptr), N,
                reinterpret_cast<cuComplex *>(c_ptr), N,
                reinterpret_cast<cuComplex *>(r_ptr), N, mode, scaling);
            num_tests++;
            if (res == 0) {
              num_passed++;
            }
          }
        }
      }
    }
  }
  CUTF_CHECK_ERROR(cudaDeviceSynchronize());

  std::printf("Result : %u / %u passed\n", num_passed, num_tests);

  cumpsgemm::destroy(cuMpSGEMM_handle);

  cutf::memory::free(a_ptr);
  cutf::memory::free(b_ptr);
  cutf::memory::free(c_ptr);
}

void gemm_tall_skinny_test(const std::size_t N, const std::size_t min_K,
                           const std::size_t max_K, const std::size_t interval,
                           const std::vector<implementation_type> &imp_list,
                           const gemm_type gemm, const bool is_seq) {
  constexpr uint64_t seed = 0;
  const std::size_t max_num_AB_elements =
      (is_seq ? N * max_K : (N << max_K)) * (gemm == gemm_type::c ? 2 : 1);
  const std::size_t num_C_elements = N * N * (gemm == gemm_type::c ? 2 : 1);
  float *a_ptr = cutf::memory::malloc<float>(max_num_AB_elements);
  float *b_ptr = cutf::memory::malloc<float>(max_num_AB_elements);
  float *c_ptr = cutf::memory::malloc<float>(num_C_elements);
  float *r_ptr = cutf::memory::malloc<float>(num_C_elements);

  auto curand_gen =
      cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_PHILOX4_32_10);
  CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*curand_gen.get(), seed));
  CUTF_CHECK_ERROR(cutf::curand::generate_normal(*curand_gen.get(), a_ptr,
                                                 max_num_AB_elements, 0, 1));
  CUTF_CHECK_ERROR(cutf::curand::generate_normal(*curand_gen.get(), b_ptr,
                                                 max_num_AB_elements, 0, 1));

  std::vector<cublasOperation_t> sgemm_ops = {CUBLAS_OP_N, CUBLAS_OP_T};
  std::vector<cublasOperation_t> cgemm_ops = {CUBLAS_OP_N, CUBLAS_OP_T,
                                              CUBLAS_OP_C};

  std::printf("## %s\n", __func__);
  std::printf("type,mode,op_A,op_B,m,n,k,throughput_in_tflops,residual,check,"
              "module_stage\n");
  unsigned num_tests = 0;
  unsigned num_passed = 0;
  auto cublas_handle_uptr = cutf::cublas::get_cublas_unique_ptr();
  cumpsgemm::handle_t cuMpSGEMM_handle;
  cumpsgemm::create(cuMpSGEMM_handle);

  std::vector<std::size_t> K_list;
  if (is_seq) {
    for (unsigned K = min_K; K <= max_K; K += interval) {
      K_list.push_back(K);
    }
  } else {
    for (unsigned K = min_K; K <= max_K; K += interval) {
      K_list.push_back(1lu << K);
    }
  }

  for (const auto imp : imp_list) {
    const auto mode = get_compute_mode(imp);
    const auto scaling = is_scaling_enabled(imp);
    if (gemm == gemm_type::s) {
      for (const auto op_A : sgemm_ops) {
        for (const auto op_B : sgemm_ops) {
          for (const auto K : K_list) {
            const auto res = sgemm_test_core(
                *cublas_handle_uptr.get(), cuMpSGEMM_handle, op_A, op_B, N, N,
                K, a_ptr, op_A == CUBLAS_OP_N ? N : K, b_ptr,
                op_B == CUBLAS_OP_N ? K : N, c_ptr, N, r_ptr, N, mode, scaling);
            num_tests++;
            if (res == 0) {
              num_passed++;
            }
          }
        }
      }
    } else if (gemm == gemm_type::c) {
      for (const auto op_A : cgemm_ops) {
        for (const auto op_B : cgemm_ops) {
          for (const auto K : K_list) {
            const auto res = sgemm_test_core(
                *cublas_handle_uptr.get(), cuMpSGEMM_handle, op_A, op_B, N, N,
                K, reinterpret_cast<cuComplex *>(a_ptr),
                op_A == CUBLAS_OP_N ? N : K,
                reinterpret_cast<cuComplex *>(b_ptr),
                op_B == CUBLAS_OP_N ? K : N,
                reinterpret_cast<cuComplex *>(c_ptr), N,
                reinterpret_cast<cuComplex *>(r_ptr), N, mode, scaling);
            num_tests++;
            if (res == 0) {
              num_passed++;
            }
          }
        }
      }
    }
  }
  CUTF_CHECK_ERROR(cudaDeviceSynchronize());

  std::printf("Result : %u / %u passed\n", num_passed, num_tests);

  cumpsgemm::destroy(cuMpSGEMM_handle);

  cutf::memory::free(a_ptr);
  cutf::memory::free(b_ptr);
  cutf::memory::free(c_ptr);
}

void gemm_strided_batch_test(const std::size_t min_N, const std::size_t max_N,
                             const std::size_t interval,
                             const std::size_t batch_count,
                             const std::vector<implementation_type> &imp_list,
                             const gemm_type gemm, const bool is_seq) {
  constexpr uint64_t seed = 0;
  const std::size_t max_num_elements =
      (is_seq ? max_N * max_N : (1lu << (2 * max_N))) *
      (gemm == gemm_type::c ? 2 : 1) * batch_count;
  float *a_ptr = cutf::memory::malloc<float>(max_num_elements);
  float *b_ptr = cutf::memory::malloc<float>(max_num_elements);
  float *c_ptr = cutf::memory::malloc<float>(max_num_elements);

  auto curand_gen =
      cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_PHILOX4_32_10);
  CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*curand_gen.get(), seed));
  CUTF_CHECK_ERROR(cutf::curand::generate_normal(*curand_gen.get(), a_ptr,
                                                 max_num_elements, 0, 1));
  CUTF_CHECK_ERROR(cutf::curand::generate_normal(*curand_gen.get(), b_ptr,
                                                 max_num_elements, 0, 1));

  std::vector<cublasOperation_t> sgemm_ops = {CUBLAS_OP_N, CUBLAS_OP_T};
  std::vector<cublasOperation_t> cgemm_ops = {CUBLAS_OP_N, CUBLAS_OP_T,
                                              CUBLAS_OP_C};

  std::vector<std::size_t> N_list;
  if (is_seq) {
    for (unsigned N = min_N; N <= max_N; N += interval) {
      N_list.push_back(N);
    }
  } else {
    for (unsigned N = min_N; N <= max_N; N += interval) {
      N_list.push_back(1lu << N);
    }
  }

  std::printf("## %s\n", __func__);
  std::printf("type,mode,op_A,op_B,m,n,k,batch_count,throughput_in_tflops,"
              "residual,check,module_stage\n");
  unsigned num_tests = 0;
  unsigned num_passed = 0;
  auto cublas_handle_uptr = cutf::cublas::get_cublas_unique_ptr();
  cumpsgemm::handle_t cuMpSGEMM_handle;
  cumpsgemm::create(cuMpSGEMM_handle);

  const auto stride = is_seq ? max_N * max_N : (1lu << (2 * max_N));

  for (const auto imp : imp_list) {
    const auto mode = get_compute_mode(imp);
    const auto scaling = is_scaling_enabled(imp);
    if (gemm == gemm_type::s) {
      for (const auto op_A : sgemm_ops) {
        for (const auto op_B : sgemm_ops) {
          for (const auto N : N_list) {
            const auto res = sgemm_strided_batch_test_core(
                *cublas_handle_uptr.get(), cuMpSGEMM_handle, op_A, op_B, N, N,
                N, a_ptr, N, stride, b_ptr, N, stride, c_ptr, N, stride,
                batch_count, mode, scaling);
            num_tests++;
            if (res == 0) {
              num_passed++;
            }
          }
        }
      }
    } else if (gemm == gemm_type::c) {
      for (const auto op_A : cgemm_ops) {
        for (const auto op_B : cgemm_ops) {
          for (const auto N : N_list) {
            const auto res = sgemm_strided_batch_test_core(
                *cublas_handle_uptr.get(), cuMpSGEMM_handle, op_A, op_B, N, N,
                N, reinterpret_cast<cuComplex *>(a_ptr), N, stride,
                reinterpret_cast<cuComplex *>(b_ptr), N, stride,
                reinterpret_cast<cuComplex *>(c_ptr), N, stride, batch_count,
                mode, scaling);
            num_tests++;
            if (res == 0) {
              num_passed++;
            }
          }
        }
      }
    }
  }
  CUTF_CHECK_ERROR(cudaDeviceSynchronize());

  std::printf("Result : %u / %u passed\n", num_passed, num_tests);

  cumpsgemm::destroy(cuMpSGEMM_handle);

  cutf::memory::free(a_ptr);
  cutf::memory::free(b_ptr);
  cutf::memory::free(c_ptr);
}

// [cuMpSGEMM LOG] cublasCgemm_v2 op=(N, T), shape=(4, 128, 65536),
// mode=TF32TCEC
void test_logged_shape(const std::string log_path) {
  std::ifstream ifs(log_path);
  if (!ifs) {
    throw std::runtime_error("No such file : " + log_path);
  }

  auto cublas_handle_uptr = cutf::cublas::get_cublas_unique_ptr();

  cumpsgemm::handle_t cuMpSGEMM_handle;
  cumpsgemm::create(cuMpSGEMM_handle);

  std::size_t num_passed = 0;
  std::size_t num_tested = 0;
  std::printf("## %s\n", __func__);
  const std::string log_prefix = "[cuMpSGEMM LOG] ";
  std::string buffer;
  while (std::getline(ifs, buffer)) {
    if (buffer.find(log_prefix) == std::string::npos) {
      continue;
    }
    buffer = buffer.substr(log_prefix.length());
    std::regex base_regex(R"((\w+) (.+), mode=(.+))");
    std::smatch base_match;

    std::string func = "";
    std::string params = "";
    std::string mode = "";
    if (std::regex_match(buffer, base_match, base_regex)) {
      func = base_match[1].str();
      params = base_match[2].str();
      mode = base_match[3].str();
    }

    if (func.length() * params.length() * mode.length() == 0) {
      continue;
    }

    cuMpSGEMM_compute_mode_t compute_mode = CUMPSGEMM_CUBLAS;
    if (mode == "FP16TC") {
      compute_mode = CUMPSGEMM_FP16TC;
    } else if (mode == "FP16TCEC") {
      compute_mode = CUMPSGEMM_FP16TCEC;
    } else if (mode == "TF32TC") {
      compute_mode = CUMPSGEMM_TF32TC;
    } else if (mode == "TF32TCEC") {
      compute_mode = CUMPSGEMM_TF32TCEC;
    } else if (mode == "FP32_SIMT") {
      compute_mode = CUMPSGEMM_FP32_SIMT;
    } else {
      throw std::runtime_error("Unknown compute mode : " + mode);
    }

    if (func == "cublasCgemm_v2" || func == "cublasSgemm_v2") {
      std::regex param_regex(
          R"(op=\((.), (.)\), shape=\((\d+), (\d+), (\d+)\))");
      std::smatch param_match;

      std::size_t m = 0, n = 0, k = 0;
      cublasOperation_t op_A, op_B;
      if (std::regex_match(params, param_match, param_regex) &&
          param_match.size() > 1) {
        op_A = param_match[1].str() == "N"
                   ? CUBLAS_OP_N
                   : (param_match[1].str() == "T" ? CUBLAS_OP_T : CUBLAS_OP_C);
        op_B = param_match[2].str() == "N"
                   ? CUBLAS_OP_N
                   : (param_match[2].str() == "T" ? CUBLAS_OP_T : CUBLAS_OP_C);
        m = std::stoul(param_match[3].str());
        n = std::stoul(param_match[4].str());
        k = std::stoul(param_match[5].str());
      } else {
        throw std::runtime_error("Failed to parse parameters : " + params);
      }

      if (m * n * k == 0) {
        throw std::runtime_error("Invalid shape : (" + std::to_string(m) +
                                 ", " + std::to_string(n) + ", " +
                                 std::to_string(k) + ")");
      }
      constexpr uint64_t seed = 0;

      const std::size_t num_e = (func == "cublasSgemm_v2" ? 1 : 2);
      float *a_ptr = cutf::memory::malloc<float>(m * k * num_e);
      float *b_ptr = cutf::memory::malloc<float>(k * n * num_e);
      float *c_ptr = cutf::memory::malloc<float>(m * n * num_e);

      auto curand_gen =
          cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_PHILOX4_32_10);
      CUTF_CHECK_ERROR(
          curandSetPseudoRandomGeneratorSeed(*curand_gen.get(), seed));
      CUTF_CHECK_ERROR(cutf::curand::generate_normal(*curand_gen.get(), a_ptr,
                                                     m * k * num_e, 0, 1));
      CUTF_CHECK_ERROR(cutf::curand::generate_normal(*curand_gen.get(), b_ptr,
                                                     k * n * num_e, 0, 1));
      int res;
      if (func == "cublasSgemm_v2") {
        res = sgemm_test_core(
            *cublas_handle_uptr.get(), cuMpSGEMM_handle, op_A, op_B, m, n, k,
            a_ptr, (op_A == CUBLAS_OP_N ? m : k), b_ptr,
            (op_B == CUBLAS_OP_N ? k : n), reinterpret_cast<float *>(0), 0,
            c_ptr, m, compute_mode);
      } else {
        res = sgemm_test_core(
            *cublas_handle_uptr.get(), cuMpSGEMM_handle, op_A, op_B, m, n, k,
            reinterpret_cast<cuComplex *>(a_ptr), (op_A == CUBLAS_OP_N ? m : k),
            reinterpret_cast<cuComplex *>(b_ptr), (op_B == CUBLAS_OP_N ? k : n),
            reinterpret_cast<cuComplex *>(0), 0,
            reinterpret_cast<cuComplex *>(c_ptr), m, compute_mode);
      }
      if (res == 0) {
        num_passed++;
      }
      num_tested++;

      cutf::memory::free(a_ptr);
      cutf::memory::free(b_ptr);
      cutf::memory::free(c_ptr);
    } else if (func == "cublasSgemmStridedBatched" ||
               func == "cublasCgemmStridedBatched") {
      std::regex param_regex(
          R"(op=\((.), (.)\), shape=\((\d+), (\d+), (\d+)\), batch=([0-9]+))");
      std::smatch param_match;

      std::size_t m = 0, n = 0, k = 0;
      std::size_t batch_size = 0;
      cublasOperation_t op_A, op_B;
      if (std::regex_match(params, param_match, param_regex) &&
          param_match.size() > 1) {
        op_A = param_match[1].str() == "N"
                   ? CUBLAS_OP_N
                   : (param_match[1].str() == "T" ? CUBLAS_OP_T : CUBLAS_OP_C);
        op_B = param_match[2].str() == "N"
                   ? CUBLAS_OP_N
                   : (param_match[2].str() == "T" ? CUBLAS_OP_T : CUBLAS_OP_C);
        m = std::stoul(param_match[3].str());
        n = std::stoul(param_match[4].str());
        k = std::stoul(param_match[5].str());
        batch_size = std::stoul(param_match[6].str());
      } else {
        throw std::runtime_error("Failed to parse parameters : " + params);
      }

      if (m * n * k * batch_size == 0) {
        throw std::runtime_error(
            "Invalid shape : (" + std::to_string(m) + ", " + std::to_string(n) +
            ", " + std::to_string(k) +
            "), batch_size = " + std::to_string(batch_size));
      }
      constexpr uint64_t seed = 0;

      const std::size_t num_e = (func == "cublasSgemmStridedBatched" ? 1 : 2);
      float *a_ptr = cutf::memory::malloc<float>(m * k * num_e * batch_size);
      float *b_ptr = cutf::memory::malloc<float>(k * n * num_e * batch_size);
      float *c_ptr = cutf::memory::malloc<float>(m * n * num_e * batch_size);

      auto curand_gen =
          cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_PHILOX4_32_10);
      CUTF_CHECK_ERROR(
          curandSetPseudoRandomGeneratorSeed(*curand_gen.get(), seed));
      CUTF_CHECK_ERROR(cutf::curand::generate_normal(
          *curand_gen.get(), a_ptr, m * k * num_e * batch_size, 0, 1));
      CUTF_CHECK_ERROR(cutf::curand::generate_normal(
          *curand_gen.get(), b_ptr, k * n * num_e * batch_size, 0, 1));
      int res;
      if (func == "cublasSgemmStridedBatched") {
        res = sgemm_strided_batch_test_core(
            *cublas_handle_uptr.get(), cuMpSGEMM_handle, op_A, op_B, m, n, k,
            a_ptr, (op_A == CUBLAS_OP_N ? m : k), m * k, b_ptr,
            (op_B == CUBLAS_OP_N ? k : n), k * n, c_ptr, m, m * n, batch_size,
            compute_mode);
      } else {
        res = sgemm_strided_batch_test_core(
            *cublas_handle_uptr.get(), cuMpSGEMM_handle, op_A, op_B, m, n, k,
            reinterpret_cast<cuComplex *>(a_ptr), (op_A == CUBLAS_OP_N ? m : k),
            m * k, reinterpret_cast<cuComplex *>(b_ptr),
            (op_B == CUBLAS_OP_N ? k : n), k * n,
            reinterpret_cast<cuComplex *>(c_ptr), m, m * n, batch_size,
            compute_mode);
      }
      if (res == 0) {
        num_passed++;
      }
      num_tested++;

      cutf::memory::free(a_ptr);
      cutf::memory::free(b_ptr);
      cutf::memory::free(c_ptr);
    }
  }
  ifs.close();
  std::printf("%lu / %lu passed\n", num_passed, num_tested);

  cumpsgemm::destroy(cuMpSGEMM_handle);
}

void gemm_exp_stats_test(const std::size_t N, const float ignore_threshold,
                         const float underflow_threshold, const gemm_type gemm,
                         const std::size_t batch_size = 1) {
  constexpr uint64_t seed = 0;
  const std::size_t max_num_elements =
      N * N * (gemm == gemm_type::c ? 2 : 1) * batch_size;
  float *a_ptr = cutf::memory::malloc<float>(max_num_elements);
  float *b_ptr = cutf::memory::malloc<float>(max_num_elements);
  float *c_ptr = cutf::memory::malloc<float>(max_num_elements);
  float *a_org_ptr = cutf::memory::malloc<float>(max_num_elements);
  float *b_org_ptr = cutf::memory::malloc<float>(max_num_elements);

  auto curand_gen =
      cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_PHILOX4_32_10);
  CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*curand_gen.get(), seed));
  CUTF_CHECK_ERROR(cutf::curand::generate_normal(
      *curand_gen.get(), a_ptr, max_num_elements, 0.f, 1.f / 65536));
  CUTF_CHECK_ERROR(cutf::curand::generate_normal(
      *curand_gen.get(), b_ptr, max_num_elements, 0.f, 1.f / 65536));
  cutf::memory::copy(a_org_ptr, a_ptr, max_num_elements);
  cutf::memory::copy(b_org_ptr, b_ptr, max_num_elements);

  std::printf("## %s\n", __func__);
  auto cublas_handle_uptr = cutf::cublas::get_cublas_unique_ptr();
  cumpsgemm::handle_t cuMpSGEMM_handle;
  cumpsgemm::create(cuMpSGEMM_handle);
  cumpsgemm::set_exp_stats_params(cuMpSGEMM_handle, ignore_threshold,
                                  underflow_threshold, 0.1f);

  std::vector<cuMpSGEMM_compute_mode_t> modes;

  modes.push_back(CUMPSGEMM_AUTO);

  // Profiler
  cutf::debug::time_breakdown::profiler profiler;

  // Exp stats of A and B
  unsigned A_exp_stats_buffer_id, B_exp_stats_buffer_id, C_exp_stats_buffer_id;
  std::pair<std::size_t, std::size_t> A_exp_stats, B_exp_stats, C_exp_stats;

  if (gemm == gemm_type::s) {
    profiler.measure("exp_stats_A", [&]() {
      cumpsgemm::exp_stats_ext(cuMpSGEMM_handle, N, N, a_ptr, N, batch_size,
                               N * N);
    });
    A_exp_stats_buffer_id =
        cumpsgemm::get_current_exp_stats_buffer_id(cuMpSGEMM_handle);
    profiler.measure("exp_stats_B", [&]() {
      cumpsgemm::exp_stats_ext(cuMpSGEMM_handle, N, N, b_ptr, N, batch_size,
                               N * N);
    });
    B_exp_stats_buffer_id =
        cumpsgemm::get_current_exp_stats_buffer_id(cuMpSGEMM_handle);
  } else {
    profiler.measure("exp_stats_A", [&]() {
      cumpsgemm::exp_stats_ext(cuMpSGEMM_handle, N, N,
                               reinterpret_cast<cuComplex *>(a_ptr), N,
                               batch_size, N * N);
    });
    A_exp_stats_buffer_id =
        cumpsgemm::get_current_exp_stats_buffer_id(cuMpSGEMM_handle);
    profiler.measure("exp_stats_B", [&]() {
      cumpsgemm::exp_stats_ext(cuMpSGEMM_handle, N, N,
                               reinterpret_cast<cuComplex *>(b_ptr), N,
                               batch_size, N * N);
    });
    B_exp_stats_buffer_id =
        cumpsgemm::get_current_exp_stats_buffer_id(cuMpSGEMM_handle);
  }
  A_exp_stats =
      cumpsgemm::get_exp_stats(cuMpSGEMM_handle, A_exp_stats_buffer_id);
  B_exp_stats =
      cumpsgemm::get_exp_stats(cuMpSGEMM_handle, B_exp_stats_buffer_id);

  const auto dynamic_launch_id =
      cumpsgemm::get_next_dynamic_launch_buffer_id(cuMpSGEMM_handle);
  cumpsgemm::set_dynamic_launch_buffer_by_exp_stats(
      cuMpSGEMM_handle, dynamic_launch_id, A_exp_stats_buffer_id,
      B_exp_stats_buffer_id);
  const auto scale_mode_AB = cumpsgemm::get_dynamic_launch_scaling_mode_AB(
      cuMpSGEMM_handle, dynamic_launch_id);

  if (gemm == gemm_type::s) {
    profiler.measure("scale_A", [&]() {
      cumpsgemm::scale_A(cuMpSGEMM_handle, A_exp_stats_buffer_id,
                         dynamic_launch_id, N, N, a_ptr, N, batch_size, N * N);
    });
    profiler.measure("scale_B", [&]() {
      cumpsgemm::scale_B(cuMpSGEMM_handle, B_exp_stats_buffer_id,
                         dynamic_launch_id, N, N, b_ptr, N, batch_size, N * N);
    });
  } else {
    profiler.measure("scale_A", [&]() {
      cumpsgemm::scale_A(
          cuMpSGEMM_handle, A_exp_stats_buffer_id, dynamic_launch_id, N, N,
          reinterpret_cast<cuComplex *>(a_ptr), N, batch_size, N * N);
    });
    profiler.measure("scale_B", [&]() {
      cumpsgemm::scale_B(
          cuMpSGEMM_handle, B_exp_stats_buffer_id, dynamic_launch_id, N, N,
          reinterpret_cast<cuComplex *>(b_ptr), N, batch_size, N * N);
    });
  }

  // Computation
  for (const auto compute_mode : modes) {
    if (gemm == gemm_type::s) {
      const float alpha = 1.0f, beta = 0.0f;
      if (batch_size == 1) {
        profiler.measure("gemm", [&]() {
          cumpsgemm::gemm(cuMpSGEMM_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                          &alpha, a_ptr, N, b_ptr, N, &beta, c_ptr, N,
                          compute_mode);
        });
      } else {
        profiler.measure("gemm", [&]() {
          cumpsgemm::gemm_stridedBatch(cuMpSGEMM_handle, CUBLAS_OP_N,
                                       CUBLAS_OP_N, N, N, N, &alpha, a_ptr, N,
                                       N * N, b_ptr, N, N * N, &beta, c_ptr, N,
                                       N * N, batch_size, compute_mode);
        });
      }
    } else {
      const cuComplex alpha = make_float2(1, 0);
      const cuComplex beta = make_float2(0, 0);
      if (batch_size == 1) {
        profiler.measure("gemm", [&]() {
          cumpsgemm::gemm(cuMpSGEMM_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                          &alpha, reinterpret_cast<const cuComplex *>(a_ptr), N,
                          reinterpret_cast<const cuComplex *>(b_ptr), N, &beta,
                          reinterpret_cast<cuComplex *>(c_ptr), N,
                          compute_mode);
        });
      } else {
        profiler.measure("gemm", [&]() {
          cumpsgemm::gemm_stridedBatch(
              cuMpSGEMM_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
              reinterpret_cast<const cuComplex *>(a_ptr), N, N * N,
              reinterpret_cast<const cuComplex *>(b_ptr), N, N * N, &beta,
              reinterpret_cast<cuComplex *>(c_ptr), N, N * N, batch_size,
              CUMPSGEMM_TF32TCEC);
        });
      }
    }

    if (gemm == gemm_type::s) {
      profiler.measure("scale_C", [&]() {
        cumpsgemm::scale_C(cuMpSGEMM_handle, A_exp_stats_buffer_id,
                           B_exp_stats_buffer_id, dynamic_launch_id, N, N,
                           c_ptr, N, batch_size, N * N);
      });
    } else {
      profiler.measure("scale_C", [&]() {
        cumpsgemm::scale_C(cuMpSGEMM_handle, A_exp_stats_buffer_id,
                           B_exp_stats_buffer_id, dynamic_launch_id, N, N,
                           reinterpret_cast<cuComplex *>(c_ptr), N, batch_size,
                           N * N);
      });
    }

    // Exp stats of C
    if (gemm == gemm_type::s) {
      cumpsgemm::exp_stats_ext(cuMpSGEMM_handle, N, N, c_ptr, N, batch_size,
                               N * N);
      C_exp_stats_buffer_id =
          cumpsgemm::get_current_exp_stats_buffer_id(cuMpSGEMM_handle);
    } else {
      cumpsgemm::exp_stats_ext(cuMpSGEMM_handle, N, N,
                               reinterpret_cast<cuComplex *>(c_ptr), N,
                               batch_size, N * N);
      C_exp_stats_buffer_id =
          cumpsgemm::get_current_exp_stats_buffer_id(cuMpSGEMM_handle);
    }
    C_exp_stats =
        cumpsgemm::get_exp_stats(cuMpSGEMM_handle, C_exp_stats_buffer_id);

    // Check
    double residual = 0;

    for (unsigned b = 0; b < batch_size; b++) {
      if (gemm == gemm_type::s) {
        const float alpha = 1.0f, beta = 0.0f;
        residual += calc_matmul_residual(
            CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha, a_org_ptr + b * N * N, N,
            b_org_ptr + b * N * N, N, beta, reinterpret_cast<float *>(0), N,
            c_ptr + b * N * N, N);
      } else {
        const cuComplex alpha = make_float2(1, 0);
        const cuComplex beta = make_float2(0, 0);
        residual += calc_matmul_residual(
            CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha,
            reinterpret_cast<cuComplex *>(a_org_ptr) + b * N * N, N,
            reinterpret_cast<cuComplex *>(b_org_ptr) + b * N * N, N, beta,
            reinterpret_cast<cuComplex *>(0), N,
            reinterpret_cast<cuComplex *>(c_ptr) + b * N * N, N);
      }
    }
    residual /= batch_size;
    const auto check = residual < error_threshold(compute_mode, N);

    std::printf(
        "# [%s:%8s, N=%lu, batch_size=%lu, u=%e, i=%e]\n"
        "  A = [underflow: %10lu / %10lu (%6.2f), max_exp=%e, buffer_id = %u "
        "-> mode=%s]\n"
        "  B = [underflow: %10lu / %10lu (%6.2f), max_exp=%e, buffer_id = %u "
        "-> mode=%s]\n"
        "  C = [underflow: %10lu / %10lu (%6.2f), max_exp=%e, buffer_id = %u]\n"
        "  Compute mode = [%s, SCALE_A=%d, SCALE_B=%d]\n"
        "  Error = [%e, %s]\n",

        (gemm == gemm_type::s ? "sgemm" : "cgemm"),
        cuMpSGEMM_get_compute_mode_string(compute_mode), N, batch_size,
        underflow_threshold, ignore_threshold,

        A_exp_stats.first, B_exp_stats.second,
        (A_exp_stats.second == 0
             ? 0.f
             : static_cast<double>(A_exp_stats.first) / A_exp_stats.second),
        cumpsgemm::get_max_exp(cuMpSGEMM_handle, A_exp_stats_buffer_id),
        A_exp_stats_buffer_id,
        cuMpSGEMM_get_compute_mode_string(
            cumpsgemm::get_exp_stats_compute_mode_level(cuMpSGEMM_handle,
                                                        A_exp_stats_buffer_id)),

        B_exp_stats.first, B_exp_stats.second,
        (B_exp_stats.second == 0
             ? 0.f
             : static_cast<double>(B_exp_stats.first) / B_exp_stats.second),
        cumpsgemm::get_max_exp(cuMpSGEMM_handle, B_exp_stats_buffer_id),
        B_exp_stats_buffer_id,
        cuMpSGEMM_get_compute_mode_string(
            cumpsgemm::get_exp_stats_compute_mode_level(cuMpSGEMM_handle,
                                                        B_exp_stats_buffer_id)),

        C_exp_stats.first, C_exp_stats.second,
        (C_exp_stats.second == 0
             ? 0.f
             : static_cast<double>(C_exp_stats.first) / C_exp_stats.second),
        cumpsgemm::get_max_exp(cuMpSGEMM_handle, C_exp_stats_buffer_id),
        C_exp_stats_buffer_id,

        cuMpSGEMM_get_compute_mode_string(
            cumpsgemm::get_dynamic_launch_gemm_compute_mode(cuMpSGEMM_handle,
                                                            dynamic_launch_id)),
        scale_mode_AB.first, scale_mode_AB.second,

        residual, check ? "OK" : "NG");
    profiler.print_result(stdout);
    std::fflush(stdout);
  }

  cumpsgemm::destroy(cuMpSGEMM_handle);
}

void exp_stats_bw_test(const unsigned min_log_N, const unsigned max_log_N,
                       const gemm_type gemm, const std::size_t batch_size = 1) {
  constexpr uint64_t seed = 0;
  const std::size_t max_num_elements =
      (1lu << (2 * max_log_N)) * (gemm == gemm_type::c ? 2 : 1) * batch_size;
  float *a_ptr = cutf::memory::malloc<float>(max_num_elements);
  float *a_org_ptr = cutf::memory::malloc<float>(max_num_elements);

  auto curand_gen =
      cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_PHILOX4_32_10);
  CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*curand_gen.get(), seed));
  CUTF_CHECK_ERROR(cutf::curand::generate_normal(
      *curand_gen.get(), a_ptr, max_num_elements, 0.f, 1.f / 65536));

  std::printf("## %s\n", __func__);
  auto cublas_handle_uptr = cutf::cublas::get_cublas_unique_ptr();
  cumpsgemm::handle_t cuMpSGEMM_handle;
  cumpsgemm::create(cuMpSGEMM_handle);
  cumpsgemm::set_exp_stats_params(cuMpSGEMM_handle, 0, 1, 0);
  cumpsgemm::enable_exp_stats_profiling(cuMpSGEMM_handle);

  std::printf("gemm,N,exp_stats_1_bw_in_gbps,exp_stats_2_bw_in_gbps,scale_bw_"
              "in_gbps\n");
  for (unsigned log_N = min_log_N; log_N <= max_log_N; log_N++) {
    cumpsgemm::reset_exp_stats_profiling(cuMpSGEMM_handle);
    const auto N = 1lu << log_N;
    const auto num_elements =
        N * N * (gemm == gemm_type::c ? 2 : 1) * batch_size;

    cutf::memory::copy(a_ptr, a_org_ptr, num_elements);

    for (unsigned i = 0; i < 100; i++) {
      if (gemm == gemm_type::s) {
        cumpsgemm::exp_stats_ext(cuMpSGEMM_handle, N, N, a_ptr, N, batch_size,
                                 N * N);
        const auto exp_stats_id =
            cumpsgemm::get_current_exp_stats_buffer_id(cuMpSGEMM_handle);
        cumpsgemm::scale_A(cuMpSGEMM_handle, exp_stats_id, 1, N, N, a_ptr, N,
                           batch_size, N * N);
      } else {
        cumpsgemm::exp_stats_ext(cuMpSGEMM_handle, N, N,
                                 reinterpret_cast<cuComplex *>(a_ptr), N,
                                 batch_size, N * N);
        const auto exp_stats_id =
            cumpsgemm::get_current_exp_stats_buffer_id(cuMpSGEMM_handle);
        cumpsgemm::scale_A(cuMpSGEMM_handle, exp_stats_id, 1, N, N,
                           reinterpret_cast<cuComplex *>(a_ptr), N, batch_size,
                           N * N);
      }
    }

#ifdef ENABLE_AUTO_MODE_PROFILING
    cumpsgemm::print_exp_stats_profiling(cuMpSGEMM_handle);
#endif
    const auto stats =
        cumpsgemm::debug::get_exp_stats_profiling_result(cuMpSGEMM_handle);

    const auto exp_stats_1_bw = 1 * num_elements * sizeof(float) *
                                stats.at("exp_stats_1").n /
                                stats.at("exp_stats_1").time_sum;
    const auto exp_stats_2_bw = 1 * num_elements * sizeof(float) *
                                stats.at("exp_stats_2").n /
                                stats.at("exp_stats_1").time_sum;
    const auto scale_bw = 2 * num_elements * sizeof(float) *
                          stats.at("scaling_AB").n /
                          stats.at("scaling_AB").time_sum;

    std::printf("%s,%lu,%e,%e,%e\n", (gemm == gemm_type::c ? "C" : "S"), N,
                exp_stats_1_bw * 1e-9, exp_stats_2_bw * 1e-9, scale_bw * 1e-9);
    std::fflush(stdout);
  }
}

void print_usage(const char *program_name) {
  std::fprintf(
      stderr,
      "Usage : %s sgemm [exp2|seq] [min_N] [max_N] [interval] [compute mode "
      "list...]\n"
      "      : %s cgemm [exp2|seq] [min_N] [max_N] [interval] [compute mode "
      "list...]\n"
      "      : %s sgemm_strided_batch [exp2|seq] [min_N] [max_N] [interval] "
      "[batch_count] [compute mode list...]\n"
      "      : %s cgemm_strided_batch [exp2|seq] [min_N] [max_N] [interval] "
      "[batch_count] [compute mode list...]\n"
      "      : %s log [/path/to/log]\n"
      "      : %s sgemm_exp_stats [N] [ignore_threshold] "
      "[underflow_threshold]\n"
      "      : %s cgemm_exp_stats [N] [ignore_threshold] "
      "[underflow_threshold]\n"
      "      : %s sgemm_strided_batch_exp_stats [N] [batch_size] "
      "[ignore_threshold] [underflow_threshold]\n"
      "      : %s cgemm_strided_batch_exp_stats [N] [batch_size] "
      "[ignore_threshold] [underflow_threshold]\n"
      "      : %s sgemm_exp_stats_bw [min_log_N] [max_log_N] [batch_size]\n"
      "      : %s cgemm_exp_stats_bw [min_log_N] [max_log_N] [batch_size]\n"
      "      : %s sgemm_tall_skinny [exp2|seq] [MN] [min_K] [max_K] [interval] "
      "[compute mode list...]\n"
      "      : %s cgemm_tall_skinny [exp2|seq] [MN] [min_K] [max_K] [interval] "
      "[compute mode list...]\n"
      "- compute mode : FP16TCEC, TF32TCEC, FP16TC, TF32TC, FP16TCEC_SCALING, "
      "CUBLAS\n",
      program_name, program_name, program_name, program_name, program_name,
      program_name, program_name, program_name, program_name, program_name,
      program_name, program_name, program_name);
  std::fflush(stderr);
}

std::vector<implementation_type>
gen_implementation_list(const char *const *argv_implementation_list_start_ptr,
                        const unsigned len) {
  std::vector<implementation_type> imp_list;
  for (unsigned i = 0; i < len; i++) {
    const std::string imp_name_str = argv_implementation_list_start_ptr[i];
    if (imp_name_str == "CUBLAS") {
      imp_list.push_back(CUBLAS);
    } else if (imp_name_str == "FP16TCEC") {
      imp_list.push_back(FP16TCEC);
    } else if (imp_name_str == "FP16TC") {
      imp_list.push_back(FP16TC);
    } else if (imp_name_str == "TF32TCEC") {
      imp_list.push_back(TF32TCEC);
    } else if (imp_name_str == "TF32TC") {
      imp_list.push_back(TF32TC);
    } else if (imp_name_str == "FP16TCEC_SCALING") {
      imp_list.push_back(FP16TCEC_SCALING);
    } else {
      std::printf("Unknown compute mode : %s\n", imp_name_str.c_str());
    }
  }
  return imp_list;
}

void print_implementation_type_list(
    const std::vector<implementation_type> &imp_list) {
  std::printf("Testing implementations: ");
  for (const auto imp : imp_list) {
    std::printf("%s ", get_implementation_type_name_str(imp).c_str());
  }
  std::printf("\n");
  std::fflush(stdout);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  const std::string command = argv[1];

  if (command == "log") {
    if (argc < 1 + 1 + 1) {
      print_usage(argv[0]);
      return 1;
    }
    test_logged_shape(argv[2]);
    return 0;
  } else if (command == "sgemm_exp_stats" || command == "cgemm_exp_stats") {
    if (argc < 1 + 1 + 3) {
      print_usage(argv[0]);
      return 1;
    }
    gemm_exp_stats_test(
        std::stoi(argv[2]), std::stof(argv[3]), std::stof(argv[4]),
        (command == "sgemm_exp_stats" ? gemm_type::s : gemm_type::c));
    return 0;
  } else if (command == "sgemm_strided_batch_exp_stats" ||
             command == "cgemm_strided_batch_exp_stats") {
    if (argc < 1 + 1 + 4) {
      print_usage(argv[0]);
      return 1;
    }
    gemm_exp_stats_test(
        std::stoi(argv[2]), std::stof(argv[4]), std::stof(argv[5]),
        (command == "sgemm_strided_batch_exp_stats" ? gemm_type::s
                                                    : gemm_type::c),
        std::stoi(argv[3]));
    return 0;
  } else if (command == "sgemm_exp_stats_bw" ||
             command == "cgemm_exp_stats_bw") {
    if (argc < 1 + 1 + 3) {
      print_usage(argv[0]);
      return 1;
    }
    exp_stats_bw_test(
        std::stoi(argv[2]), std::stoi(argv[3]),
        (command == "sgemm_exp_stats_bw" ? gemm_type::s : gemm_type::c),
        std::stoi(argv[4]));
    return 0;
  }

  if (argc < 3 ||
      (std::string(argv[2]) != "exp2" && std::string(argv[2]) != "seq")) {
    std::fprintf(stderr, "[cuMpSGEMM test] invalid argument\n");
    return 1;
  }

  const bool is_seq = std::string(argv[2]) != "exp2";

  if (command == "sgemm" || command == "cgemm") {
    if (argc < 1 + 1 + 3 + 1) {
      print_usage(argv[0]);
      return 1;
    }
    const auto imp_list = gen_implementation_list(argv + 6, argc - 6);
    print_implementation_type_list(imp_list);
    gemm_test(std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]),
              imp_list, (command == "sgemm" ? gemm_type::s : gemm_type::c),
              is_seq);
  } else if (command == "sgemm_tall_skinny" || command == "cgemm_tall_skinny") {
    if (argc < 1 + 1 + 1 + 3 + 1) {
      print_usage(argv[0]);
      return 1;
    }
    const auto imp_list = gen_implementation_list(argv + 7, argc - 7);
    print_implementation_type_list(imp_list);
    gemm_tall_skinny_test(
        std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]),
        std::stoi(argv[6]), imp_list,
        (command == "sgemm_tall_skinny" ? gemm_type::s : gemm_type::c), is_seq);
  } else if (command == "sgemm_strided_batch" ||
             command == "cgemm_strided_batch") {
    if (argc < 1 + 1 + 3 + 1 + 1) {
      print_usage(argv[0]);
      return 1;
    }
    const auto imp_list = gen_implementation_list(argv + 7, argc - 7);
    print_implementation_type_list(imp_list);
    gemm_strided_batch_test(
        std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]),
        std::stoi(argv[6]), imp_list,
        (command == "sgemm_strided_batch" ? gemm_type::s : gemm_type::c),
        is_seq);
  } else {
    print_usage(argv[0]);
    return 1;
  }
}
