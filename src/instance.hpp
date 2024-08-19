#pragma once
#include <cstdint>
#include <cumpsgemm/cumpsgemm.h>

namespace cumpsgemm {
namespace device {
template <class T> struct element_t_conv {
  using type = T;
};
template <> struct element_t_conv<float2> {
  using type = float;
};
} // namespace device
// for exp stats
using counter_t = unsigned long long int;

template <class T>
using gemm_kernel_func_t = void (*)(const int *const dynamic_mode,
                                    const std::uint32_t, const std::uint32_t,
                                    const std::uint32_t, const T,
                                    const T *const, const std::uint32_t,
                                    const T *const, const std::uint32_t,
                                    const T, T *const, const std::uint32_t);

template <class T>
using gemm_stridedBatch_kernel_func_t =
    void (*)(const int *const dynamic_mode, const std::uint32_t,
             const std::uint32_t, const std::uint32_t, const T, const T *const,
             const std::uint32_t, const std::uint64_t, const T *const,
             const std::uint32_t, const std::uint64_t, const T, T *const,
             const std::uint32_t, const std::uint64_t, const std::uint32_t);

struct gemm_module {
  void *kernel_func;

  unsigned smem_m, smem_n, smem_k;
  unsigned smem_size;
  unsigned block_size;
  unsigned num_active_blocks;
  unsigned k_per_mn;
};

// 0 is for large size matmul and (num_kernel_candidates - 1) is for small.
static constexpr unsigned num_kernel_candidates = 3;

namespace kernel_module_code {
using code_t = std::uint32_t;
constexpr code_t op_a_col_major = 0b0'0'00'00'01;
constexpr code_t op_a_row_major = 0b0'0'00'00'10;
constexpr code_t op_a_conjugate = 0b0'0'00'00'11;
constexpr code_t op_b_col_major = 0b0'0'00'01'00;
constexpr code_t op_b_row_major = 0b0'0'00'10'00;
constexpr code_t op_b_conjugate = 0b0'0'00'11'00;
constexpr code_t half = 0b0'0'00'00'00;
constexpr code_t tf32 = 0b0'0'01'00'00;
constexpr code_t simt = 0b0'0'10'00'00;
constexpr code_t with_ec = 0b0'0'00'00'00;
constexpr code_t without_ec = 0b0'1'00'00'00;
constexpr code_t s = 0b0'0'00'00'00;
constexpr code_t c = 0b1'0'00'00'00;
// ------- OR accumulation ------
constexpr code_t max_code = 0b1'1'11'11'11 + 1;
} // namespace kernel_module_code
namespace exp_stats {
struct exp_stats_handle;
} // namespace exp_stats
namespace dynamic_launch {
struct dynamic_launch_handle;
} // namespace dynamic_launch

void configure_instance_sm80(
    cumpsgemm::gemm_module gemm_module[cumpsgemm::kernel_module_code::max_code]
                                      [cumpsgemm::num_kernel_candidates],
    cumpsgemm::gemm_module
        gemm_stridedBatch_module[cumpsgemm::kernel_module_code::max_code]
                                [cumpsgemm::num_kernel_candidates],
    cumpsgemm::gemm_module
        gemm_atomic_module[cumpsgemm::kernel_module_code::max_code]);
void configure_instance_sm86(
    cumpsgemm::gemm_module gemm_module[cumpsgemm::kernel_module_code::max_code]
                                      [cumpsgemm::num_kernel_candidates],
    cumpsgemm::gemm_module
        gemm_stridedBatch_module[cumpsgemm::kernel_module_code::max_code]
                                [cumpsgemm::num_kernel_candidates],
    cumpsgemm::gemm_module
        gemm_atomic_module[cumpsgemm::kernel_module_code::max_code]);
} // namespace cumpsgemm

#define SET_GEMM_KERNEL_MODULE(module_list, io_t, tc_t, ec, op_a, op_b,        \
                               smem_m, smem_n, smem_k, frag_m, frag_n, frag_k, \
                               block_size, num_unrollings, num_stages,         \
                               pipelined, gemm_type, stage)                    \
  module_list[cumpsgemm::kernel_module_code::tc_t |                            \
              cumpsgemm::kernel_module_code::ec |                              \
              cumpsgemm::kernel_module_code::op_a_##op_a |                     \
              cumpsgemm::kernel_module_code::op_b_##op_b |                     \
              cumpsgemm::kernel_module_code::gemm_type][stage] =               \
      cumpsgemm::generate_gemm_module<                                         \
          io_t, smem_m, smem_n, smem_k, frag_m, frag_n, frag_k, block_size,    \
          num_unrollings, num_stages, cumpsgemm::op_a, cumpsgemm::op_b, tc_t,  \
          mtk::wmma::tcec::ec, pipelined>();

#define SET_GEMM_STRIDEDBATCH_KERNEL_MODULE(                                   \
    module_list, io_t, tc_t, ec, op_a, op_b, smem_m, smem_n, smem_k, frag_m,   \
    frag_n, frag_k, block_size, num_unrollings, num_stages, pipelined,         \
    gemm_type, stage)                                                          \
  module_list[cumpsgemm::kernel_module_code::tc_t |                            \
              cumpsgemm::kernel_module_code::ec |                              \
              cumpsgemm::kernel_module_code::op_a_##op_a |                     \
              cumpsgemm::kernel_module_code::op_b_##op_b |                     \
              cumpsgemm::kernel_module_code::gemm_type][stage] =               \
      cumpsgemm::generate_gemm_stridedBatch_module<                            \
          io_t, smem_m, smem_n, smem_k, frag_m, frag_n, frag_k, block_size,    \
          num_unrollings, num_stages, cumpsgemm::op_a, cumpsgemm::op_b, tc_t,  \
          mtk::wmma::tcec::ec, pipelined>();

#define SET_GEMM_ATOMIC_KERNEL_MODULE(                                         \
    module_list, io_t, tc_t, ec, op_a, op_b, smem_m, smem_n, smem_k, k_per_mn, \
    frag_m, frag_n, frag_k, block_size, num_unrollings, num_stages, pipelined, \
    gemm_type)                                                                 \
  module_list[cumpsgemm::kernel_module_code::tc_t |                            \
              cumpsgemm::kernel_module_code::ec |                              \
              cumpsgemm::kernel_module_code::op_a_##op_a |                     \
              cumpsgemm::kernel_module_code::op_b_##op_b |                     \
              cumpsgemm::kernel_module_code::gemm_type] =                      \
      cumpsgemm::generate_gemm_atomic_module<                                  \
          io_t, smem_m, smem_n, smem_k, k_per_mn, frag_m, frag_n, frag_k,      \
          block_size, num_unrollings, num_stages, cumpsgemm::op_a,             \
          cumpsgemm::op_b, tc_t, mtk::wmma::tcec::ec, pipelined>();

#define COMPILE_SGEMM_KERNEL
#define COMPILE_CGEMM_KERNEL
#define COMPILE_SGEMM_STRIDEDBATCH_KERNEL
#define COMPILE_CGEMM_STRIDEDBATCH_KERNEL
#define COMPILE_SGEMM_ATOMIC_KERNEL
#define COMPILE_CGEMM_ATOMIC_KERNEL
