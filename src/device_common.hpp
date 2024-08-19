#ifndef __CUMPGEMM_DEVICE_COMMON_HPP__
#define __CUMPGEMM_DEVICE_COMMON_HPP__
#include <mma.h>

namespace cumpsgemm {

struct col_major;
struct row_major;
struct conjugate;

namespace device {
template <class CUMPSGEMM_OP> struct layout_conv {
  using type = void;
};
template <> struct layout_conv<cumpsgemm::col_major> {
  using type = nvcuda::wmma::col_major;
};
template <> struct layout_conv<cumpsgemm::row_major> {
  using type = nvcuda::wmma::row_major;
};

// zero
template <class T> __device__ inline T zero() { return 0; }
template <> __device__ inline cuComplex zero<cuComplex>() {
  return make_cuComplex(0, 0);
}

template <class T> struct size_of {
  static constexpr unsigned value = 0;
};
template <> struct size_of<ulong2> {
  static constexpr unsigned value = 16;
};
template <> struct size_of<ulong1> {
  static constexpr unsigned value = 8;
};
template <> struct size_of<uint1> {
  static constexpr unsigned value = 4;
};
template <> struct size_of<float> {
  static constexpr unsigned value = 4;
};
template <> struct size_of<cuComplex> {
  static constexpr unsigned value = 8;
};

template <class T> __device__ inline T mul(const T a, const T alpha) {
  return a * alpha;
}
template <>
__device__ inline cuComplex mul<cuComplex>(const cuComplex a,
                                           const cuComplex alpha) {
  return make_cuComplex(a.x * alpha.x - a.y * alpha.y,
                        a.y * alpha.x + a.x * alpha.y);
}

template <class T>
__device__ T inline mad(const T a, const T alpha, const T b) {
  return a * alpha + b;
}
template <>
__device__ cuComplex inline mad<cuComplex>(const cuComplex a,
                                           const cuComplex alpha,
                                           const cuComplex b) {
  return make_cuComplex(a.x * alpha.x - a.y * alpha.y + b.x,
                        a.y * alpha.x + a.x * alpha.y + b.y);
}

template <class T> __host__ __device__ bool inline is_zero(const T &v) {
  return v == 0;
}
template <> __host__ __device__ bool inline is_zero(const cuComplex &v) {
  return v.x == 0 && v.y == 0;
}

__device__ inline float atomic_add(float *const ptr, const float a) {
  return atomicAdd(ptr, a);
}
__device__ inline cuComplex atomic_add(cuComplex *const ptr,
                                       const cuComplex a) {
  float *const px = &(ptr->x);
  float *const py = &(ptr->y);
  const auto x = ::atomicAdd(px, a.x);
  const auto y = ::atomicAdd(py, a.y);
  return make_cuComplex(x, y);
}
} // namespace device
} // namespace cumpsgemm
#endif
