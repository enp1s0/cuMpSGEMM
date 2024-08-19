#pragma once
#include "device_common.hpp"
#include <cutf/cp_async.hpp>

namespace cumpsgemm {
namespace device {
namespace detail {
template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW,
          unsigned BLOCK_SIZE, unsigned v_bit_len>
__device__ void load_core(T *const smem_ptr, const T *const dmem_ptr,
                          const unsigned ld, const unsigned start_m,
                          const unsigned start_n, const unsigned size_m,
                          const unsigned size_n) {
  if constexpr ((v_bit_len / size_of<T>::value) != 0) {
    const auto index = threadIdx.x * (v_bit_len / size_of<T>::value);
    const auto m = index % SMEM_M;
    const auto n = index / SMEM_M;
    auto smem_local_ptr = smem_ptr + (m + n * (SMEM_M + SKEW));
    auto dmem_local_ptr =
        dmem_ptr + (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;
    cutf::cp_async::cp_async<v_bit_len>(smem_local_ptr, dmem_local_ptr);

    for (unsigned offset = 1;
         offset <
         SMEM_M * SMEM_N / (BLOCK_SIZE * (v_bit_len / size_of<T>::value));
         offset++) {
      smem_local_ptr += (SMEM_M + SKEW) * (v_bit_len / size_of<T>::value) *
                        BLOCK_SIZE / SMEM_M;
      dmem_local_ptr +=
          static_cast<std::size_t>((v_bit_len / size_of<T>::value) *
                                   BLOCK_SIZE / SMEM_M) *
          ld;
      cutf::cp_async::cp_async<v_bit_len>(smem_local_ptr, dmem_local_ptr);
    }
  }
}

// Dmem loader
template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW,
          unsigned BLOCK_SIZE>
struct dmem_loader_core {
  __device__ void operator()(T *const smem_ptr, const T *const dmem_ptr,
                             const unsigned ld, const unsigned start_m,
                             const unsigned start_n, const unsigned size_m,
                             const unsigned size_n) {
    if (start_m + SMEM_M <= size_m && start_n + SMEM_N <= size_n) {
      if (ld % (16 / size_of<T>::value) == 0) {
        load_core<T, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE, 16>(
            smem_ptr, dmem_ptr, ld, start_m, start_n, size_m, size_n);
      } else if ((ld % (8 / size_of<T>::value) == 0)) {
        load_core<T, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE, 8>(
            smem_ptr, dmem_ptr, ld, start_m, start_n, size_m, size_n);
      } else {
        load_core<T, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE, 4>(
            smem_ptr, dmem_ptr, ld, start_m, start_n, size_m, size_n);
      }
    } else {
      for (unsigned offset = 0; offset < SMEM_M * SMEM_N;
           offset += BLOCK_SIZE) {
        const auto index = offset + threadIdx.x;
        const auto m = index % SMEM_M;
        const auto n = index / SMEM_M;
        const auto smem_index = m + n * (SMEM_M + SKEW);
        const auto dmem_index =
            (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;

        if ((start_m + m) < size_m && (start_n + n) < size_n) {
          cutf::cp_async::cp_async<size_of<T>::value>(&smem_ptr[smem_index],
                                                      &dmem_ptr[dmem_index]);
        } else {
          smem_ptr[smem_index] = zero<T>();
        }
        __syncwarp();
      }
    }
  }
};
} // namespace detail

template <class _Layout, class T, unsigned SMEM_M, unsigned SMEM_N,
          unsigned SKEW, unsigned BLOCK_SIZE>
struct dmem_loader {
  using Layout = _Layout;
  __device__ void operator()(T *const smem_ptr, const T *const dmem_ptr,
                             const unsigned ld, const unsigned start_m,
                             const unsigned start_n, const unsigned size_m,
                             const unsigned size_n) {
    detail::dmem_loader_core<T, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE>{}(
        smem_ptr, dmem_ptr, ld, start_m, start_n, size_m, size_n);
  }
};

template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW,
          unsigned BLOCK_SIZE>
struct dmem_loader<cumpsgemm::row_major, T, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE> {
  using Layout = cumpsgemm::row_major;
  __device__ void operator()(T *const smem_ptr, const T *const dmem_ptr,
                             const unsigned ld, const unsigned start_m,
                             const unsigned start_n, const unsigned size_m,
                             const unsigned size_n) {
    detail::dmem_loader_core<T, SMEM_N, SMEM_M, SKEW, BLOCK_SIZE>{}(
        smem_ptr, dmem_ptr, ld, start_n, start_m, size_n, size_m);
  }
};

namespace detail {
template <unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, unsigned BLOCK_SIZE>
struct dmem_loader_conj_core {
  __device__ void operator()(cuComplex *const smem_ptr,
                             const cuComplex *const dmem_ptr, const unsigned ld,
                             const unsigned start_m, const unsigned start_n,
                             const unsigned size_m, const unsigned size_n) {
    if (start_m + SMEM_M < size_m && start_n + SMEM_N < size_n) {
      for (unsigned offset = 0; offset < SMEM_M * SMEM_N;
           offset += BLOCK_SIZE) {
        const auto index = offset + threadIdx.x;
        const auto m = index % SMEM_M;
        const auto n = index / SMEM_M;
        const auto smem_index = m + n * (SMEM_M + SKEW);
        const auto dmem_index =
            (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;
        const auto v = dmem_ptr[dmem_index];
        smem_ptr[smem_index] = make_cuComplex(v.x, -v.y);
      }
    } else {
      for (unsigned offset = 0; offset < SMEM_M * SMEM_N;
           offset += BLOCK_SIZE) {
        const auto index = offset + threadIdx.x;
        const auto m = index % SMEM_M;
        const auto n = index / SMEM_M;
        const auto smem_index = m + n * (SMEM_M + SKEW);
        const auto dmem_index =
            (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;

        auto v = zero<cuComplex>();
        if ((start_m + m) < size_m && (start_n + n) < size_n) {
          const auto w = dmem_ptr[dmem_index];
          v = make_cuComplex(w.x, -w.y);
        }
        __syncwarp();
        smem_ptr[smem_index] = v;
      }
    }
  }
};
} // namespace detail

template <unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, unsigned BLOCK_SIZE>
struct dmem_loader<cumpsgemm::conjugate, cuComplex, SMEM_M, SMEM_N, SKEW,
                   BLOCK_SIZE> {
  using Layout = cumpsgemm::row_major;
  __device__ void operator()(cuComplex *const smem_ptr,
                             const cuComplex *const dmem_ptr, const unsigned ld,
                             const unsigned start_m, const unsigned start_n,
                             const unsigned size_m, const unsigned size_n) {
    detail::dmem_loader_conj_core<SMEM_N, SMEM_M, SKEW, BLOCK_SIZE>{}(
        smem_ptr, dmem_ptr, ld, start_n, start_m, size_n, size_m);
  }
};

template <unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, unsigned BLOCK_SIZE>
struct dmem_loader<cumpsgemm::conjugate, float, SMEM_M, SMEM_N, SKEW,
                   BLOCK_SIZE> {
  using Layout = cumpsgemm::col_major;
  __device__ void operator()(float *const, const float *const, const unsigned,
                             const unsigned, const unsigned, const unsigned,
                             const unsigned) {
    // Do nothing, only for suppressing compilation error.
  }
};

namespace detail {
template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW,
          unsigned BLOCK_SIZE, class VEC_T, bool BETA>
__device__ void dmem_store_core(T *const dmem_ptr, const unsigned ld,
                                const unsigned start_m, const unsigned start_n,
                                const unsigned size_m, const unsigned size_n,
                                const T *const smem_ptr, const T alpha,
                                const T beta) {
  constexpr unsigned v_bit_len = size_of<VEC_T>::value;
  if constexpr ((v_bit_len / size_of<T>::value) != 0) {
    const auto index = threadIdx.x * (v_bit_len / size_of<T>::value);
    const auto m = index % SMEM_M;
    const auto n = index / SMEM_M;
    auto smem_local_ptr = smem_ptr + (m + n * (SMEM_M + SKEW));
    auto dmem_local_ptr =
        dmem_ptr + (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;

    auto v = *reinterpret_cast<const VEC_T *>(smem_local_ptr);
    auto dv = *reinterpret_cast<const VEC_T *>(dmem_local_ptr);
    for (unsigned i = 0; i < v_bit_len / size_of<T>::value; i++) {
      if constexpr (BETA) {
        reinterpret_cast<T *>(&v)[i] =
            mad(reinterpret_cast<T *>(&v)[i], alpha,
                mul(beta, reinterpret_cast<T *>(&dv)[i]));
      } else {
        reinterpret_cast<T *>(&v)[i] = mul(reinterpret_cast<T *>(&v)[i], alpha);
      }
    }
    *reinterpret_cast<VEC_T *>(dmem_local_ptr) = v;

    for (unsigned offset = 1;
         offset <
         SMEM_M * SMEM_N / (BLOCK_SIZE * (v_bit_len / size_of<T>::value));
         offset++) {
      smem_local_ptr += (SMEM_M + SKEW) * (v_bit_len / size_of<T>::value) *
                        BLOCK_SIZE / SMEM_M;
      dmem_local_ptr +=
          static_cast<std::size_t>((v_bit_len / size_of<T>::value) *
                                   BLOCK_SIZE / SMEM_M) *
          ld;

      auto v = *reinterpret_cast<const VEC_T *>(smem_local_ptr);
      auto dv = *reinterpret_cast<const VEC_T *>(dmem_local_ptr);
      for (unsigned i = 0; i < v_bit_len / size_of<T>::value; i++) {
        if constexpr (BETA) {
          reinterpret_cast<T *>(&v)[i] =
              mad(reinterpret_cast<T *>(&v)[i], alpha,
                  mul(beta, reinterpret_cast<T *>(&dv)[i]));
        } else {
          reinterpret_cast<T *>(&v)[i] =
              mul(reinterpret_cast<T *>(&v)[i], alpha);
        }
      }
      *reinterpret_cast<VEC_T *>(dmem_local_ptr) = v;
    }
  }
}

template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW,
          unsigned BLOCK_SIZE, class VEC_T>
__device__ void
dmem_atomic_store_core(T *const dmem_ptr, const unsigned ld,
                       const unsigned start_m, const unsigned start_n,
                       const unsigned size_m, const unsigned size_n,
                       const T *const smem_ptr, const T alpha) {
  constexpr unsigned v_bit_len = size_of<VEC_T>::value;
  if constexpr ((v_bit_len / size_of<T>::value) != 0) {
    const auto index = threadIdx.x * (v_bit_len / size_of<T>::value);
    const auto m = index % SMEM_M;
    const auto n = index / SMEM_M;
    auto smem_local_ptr = smem_ptr + (m + n * (SMEM_M + SKEW));
    auto dmem_local_ptr =
        dmem_ptr + (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;

    auto v = *reinterpret_cast<const VEC_T *>(smem_local_ptr);
    for (unsigned i = 0; i < v_bit_len / size_of<T>::value; i++) {
      cumpsgemm::device::atomic_add(dmem_local_ptr + i,
                                    mul(reinterpret_cast<T *>(&v)[i], alpha));
    }

    for (unsigned offset = 1;
         offset <
         SMEM_M * SMEM_N / (BLOCK_SIZE * (v_bit_len / size_of<T>::value));
         offset++) {
      smem_local_ptr += (SMEM_M + SKEW) * (v_bit_len / size_of<T>::value) *
                        BLOCK_SIZE / SMEM_M;
      dmem_local_ptr +=
          static_cast<std::size_t>((v_bit_len / size_of<T>::value) *
                                   BLOCK_SIZE / SMEM_M) *
          ld;

      auto v = *reinterpret_cast<const VEC_T *>(smem_local_ptr);
      for (unsigned i = 0; i < v_bit_len / size_of<T>::value; i++) {
        cumpsgemm::device::atomic_add(dmem_local_ptr + i,
                                      mul(reinterpret_cast<T *>(&v)[i], alpha));
      }
    }
  }
}
} // namespace detail

template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW,
          unsigned BLOCK_SIZE>
struct dmem_storer {
  __device__ void operator()(T *const dmem_ptr, const unsigned ld,
                             const unsigned start_m, const unsigned start_n,
                             const unsigned size_m, const unsigned size_n,
                             const T *const smem_ptr, const T alpha,
                             const T beta) {
    if (is_zero(beta)) {
      if (start_m + SMEM_M <= size_m && start_n + SMEM_N <= size_n) {
        if (ld % (16 / size_of<T>::value) == 0) {
          detail::dmem_store_core<T, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE, ulong2,
                                  false>(dmem_ptr, ld, start_m, start_n, size_m,
                                         size_n, smem_ptr, alpha, beta);
        } else if ((ld % (8 / size_of<T>::value) == 0)) {
          detail::dmem_store_core<T, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE, ulong1,
                                  false>(dmem_ptr, ld, start_m, start_n, size_m,
                                         size_n, smem_ptr, alpha, beta);
        } else {
          detail::dmem_store_core<T, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE, uint1,
                                  false>(dmem_ptr, ld, start_m, start_n, size_m,
                                         size_n, smem_ptr, alpha, beta);
        }
      } else {
        const auto index = threadIdx.x;
        const auto m = index % SMEM_M;
        auto n = index / SMEM_M;
        auto smem_index = m + n * (SMEM_M + SKEW);
        auto dmem_index =
            (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;

        for (unsigned offset = 0; offset < SMEM_M * SMEM_N;
             offset += BLOCK_SIZE) {
          if ((start_m + m) < size_m && (start_n + n) < size_n) {
            dmem_ptr[dmem_index] = mul(smem_ptr[smem_index], alpha);
          }
          n += (BLOCK_SIZE / SMEM_M);

          smem_index += (SMEM_M + SKEW) * (BLOCK_SIZE / SMEM_M);
          dmem_index += ld * (BLOCK_SIZE / SMEM_M);
          __syncwarp();
        }
      }
    } else {
      if (start_m + SMEM_M < size_m && start_n + SMEM_N < size_n) {
        if (ld % (16 / size_of<T>::value) == 0) {
          detail::dmem_store_core<T, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE, ulong2,
                                  true>(dmem_ptr, ld, start_m, start_n, size_m,
                                        size_n, smem_ptr, alpha, beta);
        } else if ((ld % (8 / size_of<T>::value) == 0)) {
          detail::dmem_store_core<T, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE, ulong1,
                                  true>(dmem_ptr, ld, start_m, start_n, size_m,
                                        size_n, smem_ptr, alpha, beta);
        } else {
          detail::dmem_store_core<T, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE, uint1,
                                  true>(dmem_ptr, ld, start_m, start_n, size_m,
                                        size_n, smem_ptr, alpha, beta);
        }
      } else {
        for (unsigned offset = 0; offset < SMEM_M * SMEM_N;
             offset += BLOCK_SIZE) {
          const auto index = offset + threadIdx.x;
          const auto m = index % SMEM_M;
          const auto n = index / SMEM_M;
          const auto smem_index = m + n * (SMEM_M + SKEW);
          const auto dmem_index =
              (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;

          if ((start_m + m) < size_m && (start_n + n) < size_n) {
            dmem_ptr[dmem_index] = mad(smem_ptr[smem_index], alpha,
                                       mul(dmem_ptr[dmem_index], beta));
          }
          __syncwarp();
        }
      }
    }
  }
};

template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW,
          unsigned BLOCK_SIZE>
struct dmem_atomic_storer {
  __device__ void operator()(T *const dmem_ptr, const unsigned ld,
                             const unsigned start_m, const unsigned start_n,
                             const unsigned size_m, const unsigned size_n,
                             const T *const smem_ptr, const T alpha, const T) {
    if (start_m + SMEM_M <= size_m && start_n + SMEM_N <= size_n) {
      if (ld % (16 / size_of<T>::value) == 0) {
        detail::dmem_atomic_store_core<T, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE,
                                       ulong2>(dmem_ptr, ld, start_m, start_n,
                                               size_m, size_n, smem_ptr, alpha);
      } else if ((ld % (8 / size_of<T>::value) == 0)) {
        detail::dmem_atomic_store_core<T, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE,
                                       ulong1>(dmem_ptr, ld, start_m, start_n,
                                               size_m, size_n, smem_ptr, alpha);
      } else {
        detail::dmem_atomic_store_core<T, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE,
                                       uint1>(dmem_ptr, ld, start_m, start_n,
                                              size_m, size_n, smem_ptr, alpha);
      }
    } else {
      const auto index = threadIdx.x;
      const auto m = index % SMEM_M;
      auto n = index / SMEM_M;
      auto smem_index = m + n * (SMEM_M + SKEW);
      auto dmem_index =
          (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;

      for (unsigned offset = 0; offset < SMEM_M * SMEM_N;
           offset += BLOCK_SIZE) {
        if ((start_m + m) < size_m && (start_n + n) < size_n) {
          cumpsgemm::device::atomic_add(&dmem_ptr[dmem_index],
                                        mul(smem_ptr[smem_index], alpha));
        }
        n += (BLOCK_SIZE / SMEM_M);

        smem_index += (SMEM_M + SKEW) * (BLOCK_SIZE / SMEM_M);
        dmem_index += ld * (BLOCK_SIZE / SMEM_M);
        __syncwarp();
      }
    }
  }
};
} // namespace device
} // namespace cumpsgemm
