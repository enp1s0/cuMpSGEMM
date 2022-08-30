#ifndef __CUMPGEMM_INTERNAL_HPP__
#define __CUMPGEMM_INTERNAL_HPP__
#include "device_common.hpp"
#include "device_tcec_wrapper.hpp"
#include "handle.hpp"

namespace cumpsgemm {
// prototype declaration

// GEMM Params
constexpr unsigned SMEM_M = 64;
constexpr unsigned SMEM_N = 64;
constexpr unsigned SMEM_K = 32;
constexpr unsigned FRAG_M = 32;
constexpr unsigned FRAG_N = 32;
constexpr unsigned FRAG_K = 16;
constexpr unsigned BLOCK_SIZE = 128;
} // namespace cumpsgemm
#endif
