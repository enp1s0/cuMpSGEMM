#include "cumpsgemm_kernel.cuh"
#include "handle.hpp"
#include "instance.hpp"

#define SET_GEMM_SIMT_KERNEL_MODULE(module_list, io_t, tc_t, ec, op_a, op_b,   \
                                    smem_m, smem_n, smem_k, frag_m, frag_n,    \
                                    frag_k, block_size, num_unrollings,        \
                                    num_stages, pipelined, gemm_type, stage)   \
  module_list[cumpsgemm::kernel_module_code::tc_t |                            \
              cumpsgemm::kernel_module_code::ec |                              \
              cumpsgemm::kernel_module_code::op_a_##op_a |                     \
              cumpsgemm::kernel_module_code::op_b_##op_b |                     \
              cumpsgemm::kernel_module_code::gemm_type][stage] =               \
      cumpsgemm::generate_gemm_module<                                         \
          io_t, smem_m, smem_n, smem_k, frag_m, frag_n, frag_k, block_size,    \
          num_unrollings, num_stages, cumpsgemm::op_a, cumpsgemm::op_b,        \
          mtk::wmma::tcec::op_simt, mtk::wmma::tcec::ec, pipelined>();

#define SET_GEMM_SIMT_STRIDEDBATCH_KERNEL_MODULE(                              \
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
          num_unrollings, num_stages, cumpsgemm::op_a, cumpsgemm::op_b,        \
          mtk::wmma::tcec::op_simt, mtk::wmma::tcec::ec, pipelined>();

void cumpsgemm::configure_instance_simt(
    cumpsgemm::gemm_module gemm_module[cumpsgemm::kernel_module_code::max_code]
                                      [cumpsgemm::num_kernel_candidates],
    cumpsgemm::gemm_module
        gemm_stridedBatch_module[cumpsgemm::kernel_module_code::max_code]
                                [cumpsgemm::num_kernel_candidates]) {
  for (unsigned i = 0; i < cumpsgemm::num_kernel_candidates; i++) {
    SET_GEMM_SIMT_KERNEL_MODULE(gemm_module, float, simt, without_ec, col_major,
                                col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2,
                                false, s, i);
    SET_GEMM_SIMT_KERNEL_MODULE(gemm_module, float, simt, without_ec, col_major,
                                row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2,
                                false, s, i);
    SET_GEMM_SIMT_KERNEL_MODULE(gemm_module, float, simt, without_ec, row_major,
                                col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2,
                                false, s, i);
    SET_GEMM_SIMT_KERNEL_MODULE(gemm_module, float, simt, without_ec, row_major,
                                row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2,
                                false, s, i);

    SET_GEMM_SIMT_KERNEL_MODULE(gemm_module, cuComplex, simt, without_ec,
                                col_major, col_major, 128, 64, 32, 32, 32, 32,
                                128, 1, 2, false, c, i);
    SET_GEMM_SIMT_KERNEL_MODULE(gemm_module, cuComplex, simt, without_ec,
                                col_major, row_major, 128, 64, 32, 32, 32, 32,
                                128, 1, 2, false, c, i);
    SET_GEMM_SIMT_KERNEL_MODULE(gemm_module, cuComplex, simt, without_ec,
                                col_major, conjugate, 128, 64, 32, 32, 32, 32,
                                128, 1, 2, false, c, i);
    SET_GEMM_SIMT_KERNEL_MODULE(gemm_module, cuComplex, simt, without_ec,
                                row_major, col_major, 128, 64, 32, 32, 32, 32,
                                128, 1, 2, false, c, i);
    SET_GEMM_SIMT_KERNEL_MODULE(gemm_module, cuComplex, simt, without_ec,
                                row_major, row_major, 128, 64, 32, 32, 32, 32,
                                128, 1, 2, false, c, i);
    SET_GEMM_SIMT_KERNEL_MODULE(gemm_module, cuComplex, simt, without_ec,
                                row_major, conjugate, 128, 64, 32, 32, 32, 32,
                                128, 1, 2, false, c, i);
    SET_GEMM_SIMT_KERNEL_MODULE(gemm_module, cuComplex, simt, without_ec,
                                conjugate, col_major, 128, 64, 32, 32, 32, 32,
                                128, 1, 2, false, c, i);
    SET_GEMM_SIMT_KERNEL_MODULE(gemm_module, cuComplex, simt, without_ec,
                                conjugate, row_major, 128, 64, 32, 32, 32, 32,
                                128, 1, 2, false, c, i);
    SET_GEMM_SIMT_KERNEL_MODULE(gemm_module, cuComplex, simt, without_ec,
                                conjugate, conjugate, 128, 64, 32, 32, 32, 32,
                                128, 1, 2, false, c, i);

    SET_GEMM_SIMT_STRIDEDBATCH_KERNEL_MODULE(
        gemm_stridedBatch_module, float, simt, without_ec, col_major, col_major,
        128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, i);
    SET_GEMM_SIMT_STRIDEDBATCH_KERNEL_MODULE(
        gemm_stridedBatch_module, float, simt, without_ec, col_major, row_major,
        128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, i);
    SET_GEMM_SIMT_STRIDEDBATCH_KERNEL_MODULE(
        gemm_stridedBatch_module, float, simt, without_ec, row_major, col_major,
        128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, i);
    SET_GEMM_SIMT_STRIDEDBATCH_KERNEL_MODULE(
        gemm_stridedBatch_module, float, simt, without_ec, row_major, row_major,
        128, 64, 32, 32, 32, 32, 128, 1, 2, false, s, i);

    SET_GEMM_SIMT_STRIDEDBATCH_KERNEL_MODULE(
        gemm_stridedBatch_module, cuComplex, simt, without_ec, col_major,
        col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, c, i);
    SET_GEMM_SIMT_STRIDEDBATCH_KERNEL_MODULE(
        gemm_stridedBatch_module, cuComplex, simt, without_ec, col_major,
        row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, c, i);
    SET_GEMM_SIMT_STRIDEDBATCH_KERNEL_MODULE(
        gemm_stridedBatch_module, cuComplex, simt, without_ec, col_major,
        conjugate, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, c, i);
    SET_GEMM_SIMT_STRIDEDBATCH_KERNEL_MODULE(
        gemm_stridedBatch_module, cuComplex, simt, without_ec, row_major,
        col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, c, i);
    SET_GEMM_SIMT_STRIDEDBATCH_KERNEL_MODULE(
        gemm_stridedBatch_module, cuComplex, simt, without_ec, row_major,
        row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, c, i);
    SET_GEMM_SIMT_STRIDEDBATCH_KERNEL_MODULE(
        gemm_stridedBatch_module, cuComplex, simt, without_ec, row_major,
        conjugate, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, c, i);
    SET_GEMM_SIMT_STRIDEDBATCH_KERNEL_MODULE(
        gemm_stridedBatch_module, cuComplex, simt, without_ec, conjugate,
        col_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, c, i);
    SET_GEMM_SIMT_STRIDEDBATCH_KERNEL_MODULE(
        gemm_stridedBatch_module, cuComplex, simt, without_ec, conjugate,
        row_major, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, c, i);
    SET_GEMM_SIMT_STRIDEDBATCH_KERNEL_MODULE(
        gemm_stridedBatch_module, cuComplex, simt, without_ec, conjugate,
        conjugate, 128, 64, 32, 32, 32, 32, 128, 1, 2, false, c, i);
  }
}
