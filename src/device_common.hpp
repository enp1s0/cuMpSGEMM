#ifndef __CUMPGEMM_DEVICE_COMMON_HPP__
#define __CUMPGEMM_DEVICE_COMMON_HPP__
#include <mma.h>

namespace cumpsgemm {

struct col_major;
struct row_major;
struct conjugate;

namespace device {

template <class CUMPSGEMM_OP>
struct layout_conv {using type = void;};
template <> struct layout_conv<cumpsgemm::col_major> {using type = nvcuda::wmma::col_major;};
template <> struct layout_conv<cumpsgemm::row_major> {using type = nvcuda::wmma::row_major;};
} // namespace device
} // namespace cumpsgemm
#endif
