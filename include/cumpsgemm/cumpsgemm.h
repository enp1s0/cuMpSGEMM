#ifndef __CUMPSGEMM_H__
#define __CUMPSGEMM_H__
#include "detail/common.h"
#include <cstdint>
#include <cublas_v2.h>

extern "C" cublasStatus_t cuMpSGEMM_create(cuMpSGEMM_handle_t *const handle);

extern "C" cublasStatus_t cuMpSGEMM_destroy(cuMpSGEMM_handle_t handle);

extern "C" cublasStatus_t cuMpSGEMM_set_stream(cuMpSGEMM_handle_t handle,
                                               const cudaStream_t cuda_stream);

extern "C" const char *
cuMpSGEMM_get_compute_mode_string(const cuMpSGEMM_compute_mode_t mode);

// User defined function
extern "C" cuMpSGEMM_compute_mode_t cuMpSGEMM_get_compute_mode(
    const char *const func_name, cublasHandle_t const cublas_handle,
    const cublasOperation_t op_A, const cublasOperation_t op_B,
    const unsigned m, const unsigned n, const unsigned k);

extern "C" cublasStatus_t
cuMpSGEMM_sgemm(cuMpSGEMM_handle_t handle, const cublasOperation_t op_A,
                const cublasOperation_t op_B, const uint64_t m,
                const uint64_t n, const uint64_t k, const float *alpha,
                const float *const a_dmem_ptr, const uint64_t lda,
                const float *const b_dmem_ptr, const uint64_t ldb,
                const float *beta, float *const c_dmem_ptr, const uint64_t ldc,
                const cuMpSGEMM_compute_mode_t compute_mode);

extern "C" cublasStatus_t cuMpSGEMM_cgemm(
    cuMpSGEMM_handle_t handle, const cublasOperation_t op_A,
    const cublasOperation_t op_B, const uint64_t m, const uint64_t n,
    const uint64_t k, const cuComplex *alpha, const cuComplex *const a_dmem_ptr,
    const uint64_t lda, const cuComplex *const b_dmem_ptr, const uint64_t ldb,
    const cuComplex *beta, cuComplex *const c_dmem_ptr, const uint64_t ldc,
    const cuMpSGEMM_compute_mode_t compute_mode);

extern "C" cublasStatus_t cuMpSGEMM_sgemm_strided_batch(
    cuMpSGEMM_handle_t handle, const cublasOperation_t op_A,
    const cublasOperation_t op_B, const uint64_t m, const uint64_t n,
    const uint64_t k, const float *alpha, const float *const a_dmem_ptr,
    const uint64_t lda, const uint64_t stridea, const float *const b_dmem_ptr,
    const uint64_t ldb, const uint64_t strideb, const float *beta,
    float *const c_dmem_ptr, const uint64_t ldc, const uint64_t stridec,
    const uint64_t batch_count, const cuMpSGEMM_compute_mode_t compute_mode);

extern "C" cublasStatus_t cuMpSGEMM_cgemm_strided_batch(
    cuMpSGEMM_handle_t handle, const cublasOperation_t op_A,
    const cublasOperation_t op_B, const uint64_t m, const uint64_t n,
    const uint64_t k, const cuComplex *alpha, const cuComplex *const a_dmem_ptr,
    const uint64_t lda, const uint64_t stridea,
    const cuComplex *const b_dmem_ptr, const uint64_t ldb,
    const uint64_t strideb, const cuComplex *beta, cuComplex *const c_dmem_ptr,
    const uint64_t ldc, const uint64_t stridec, const uint64_t batch_count,
    const cuMpSGEMM_compute_mode_t compute_mode);

#endif
