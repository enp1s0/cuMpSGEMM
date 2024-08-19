#ifndef __CUMPGEMM_TCEC_HPP__
#define __CUMPGEMM_TCEC_HPP__
#include <wmma_extension/tcec/complex.hpp>

#include "device_common.hpp"

namespace cumpsgemm {
namespace device {
namespace detail {
template <class Use> struct use_layout_conv {
  using type = void;
};
template <> struct use_layout_conv<nvcuda::wmma::matrix_a> {
  using type = nvcuda::wmma::row_major;
};
template <> struct use_layout_conv<nvcuda::wmma::matrix_b> {
  using type = nvcuda::wmma::col_major;
};
} // namespace detail
// fragment
template <class T, class Use, unsigned M, unsigned N, unsigned K, class Layout,
          class TC_T, class EC>
struct tc_fragment {
  using frag_t = mtk::wmma::tcec::fragment<
      Use, M, N, K, TC_T, typename detail::use_layout_conv<Use>::type,
      typename mtk::wmma::tcec::default_policy<TC_T, EC,
                                               mtk::wmma::tcec::op_mma>::type>;
  frag_t frag;
};

template <class T, class Use, unsigned M, unsigned N, unsigned K, class Layout>
struct tc_fragment<T, Use, M, N, K, Layout, mtk::wmma::tcec::op_simt,
                   mtk::wmma::tcec::op_without_error_correction> {
  using frag_t = mtk::wmma::tcec::fragment<
      Use, M, N, K, float, typename detail::use_layout_conv<Use>::type,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::op_without_error_correction,
          mtk::wmma::tcec::op_simt>::type>;
  frag_t frag;
};

template <class Use, unsigned M, unsigned N, unsigned K, class Layout,
          class TC_T, class EC>
struct tc_fragment<cuComplex, Use, M, N, K, Layout, TC_T, EC> {
  using frag_t = mtk::wmma::tcec::fragment_complex<
      Use, M, N, K, TC_T, typename detail::use_layout_conv<Use>::type,
      typename mtk::wmma::tcec::default_policy<TC_T, EC,
                                               mtk::wmma::tcec::op_mma>::type>;
  frag_t frag;
};

template <class Use, unsigned M, unsigned N, unsigned K, class Layout>
struct tc_fragment<cuComplex, Use, M, N, K, Layout, mtk::wmma::tcec::op_simt,
                   mtk::wmma::tcec::op_without_error_correction> {
  using frag_t = mtk::wmma::tcec::fragment_complex<
      Use, M, N, K, float, typename detail::use_layout_conv<Use>::type,
      typename mtk::wmma::tcec::default_policy<
          float, mtk::wmma::tcec::op_without_error_correction,
          mtk::wmma::tcec::op_simt>::type>;
  frag_t frag;
};

// fill_zero
template <class MEM_T, class Use, unsigned M, unsigned N, unsigned K,
          class TC_T, class Layout, class EC>
__device__ void
fill_zero(tc_fragment<MEM_T, Use, M, N, K, Layout, TC_T, EC> &frag) {
  mtk::wmma::tcec::fill_zero(frag.frag);
}

// fragment loader
template <class MEM_T, class Use, unsigned M, unsigned N, unsigned K,
          class TC_T, class Layout, class EC>
__device__ void
load_matrix(tc_fragment<MEM_T, Use, M, N, K, Layout, TC_T, EC> &frag,
            const MEM_T *const ptr, const uint64_t ldm) {
  mtk::wmma::tcec::load_matrix_sync<
      typename cumpsgemm::device::layout_conv<Layout>::type>(frag.frag, ptr,
                                                             ldm, false);
}

// fragment storer
template <class MEM_T, class Use, unsigned M, unsigned N, unsigned K,
          class TC_T, class EC>
__device__ void
store_matrix(MEM_T *const ptr,
             tc_fragment<MEM_T, Use, M, N, K, void, TC_T, EC> &frag,
             const uint64_t ldm) {
  mtk::wmma::tcec::store_matrix_sync<nvcuda::wmma::col_major>(ptr, frag.frag,
                                                              ldm, false);
}

// mma
template <class MEM_T, class OP_A, class OP_B, unsigned M, unsigned N,
          unsigned K, class TC_T, class EC>
__device__ void
mma(tc_fragment<MEM_T, nvcuda::wmma::accumulator, M, N, K, void, TC_T, EC>
        &frag_d,
    const tc_fragment<MEM_T, nvcuda::wmma::matrix_a, M, N, K, OP_A, TC_T, EC>
        &frag_a,
    const tc_fragment<MEM_T, nvcuda::wmma::matrix_b, M, N, K, OP_B, TC_T, EC>
        &frag_b,
    const tc_fragment<MEM_T, nvcuda::wmma::accumulator, M, N, K, void, TC_T, EC>
        &frag_c) {
  mtk::wmma::tcec::mma_sync(frag_d.frag, frag_a.frag, frag_b.frag, frag_c.frag);
}

} // namespace device
} // namespace cumpsgemm
#endif
