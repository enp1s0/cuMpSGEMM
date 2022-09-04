#pragma once
#include <cutf/cp_async.hpp>
#include "device_common.hpp"

namespace cumpsgemm {
namespace device {
// Dmem loader
template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, unsigned BLOCK_SIZE>
struct dmem_loader_core {
	__device__ void operator() (
			T* const smem_ptr,
			const T* const dmem_ptr,
			const unsigned ld,
			const unsigned start_m,
			const unsigned start_n,
			const unsigned size_m,
			const unsigned size_n
			) {
		if (start_m + SMEM_M < size_m && start_n + SMEM_N < size_n) {
			if (ld % (16 / size_of<T>::value) == 0) {
				constexpr unsigned v_bit_len = 16;
				const auto index = threadIdx.x * (v_bit_len / size_of<T>::value);
				const auto m = index % SMEM_M;
				const auto n = index / SMEM_M;
				auto smem_index = m + n * (SMEM_M + SKEW);
				auto dmem_index = (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;
				cutf::cp_async::cp_async<v_bit_len>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);

				for (unsigned offset = 1; offset < SMEM_M * SMEM_N / (BLOCK_SIZE * (v_bit_len/ size_of<T>::value)); offset++) {
					smem_index += (SMEM_M + SKEW) * (v_bit_len / size_of<T>::value) * BLOCK_SIZE / SMEM_M;
					dmem_index += static_cast<std::size_t>((v_bit_len / size_of<T>::value) * BLOCK_SIZE / SMEM_M) * ld;
					cutf::cp_async::cp_async<v_bit_len>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				}
			} else if ((ld % (8 / size_of<T>::value) == 0)) {
				constexpr unsigned v_bit_len = 8;
				const auto index = threadIdx.x * (v_bit_len / size_of<T>::value);
				const auto m = index % SMEM_M;
				const auto n = index / SMEM_M;
				auto smem_index = m + n * (SMEM_M + SKEW);
				auto dmem_index = (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;
				cutf::cp_async::cp_async<v_bit_len>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);

				for (unsigned offset = 1; offset < SMEM_M * SMEM_N / (BLOCK_SIZE * (v_bit_len/ size_of<T>::value)); offset++) {
					smem_index += (SMEM_M + SKEW) * (v_bit_len / size_of<T>::value) * BLOCK_SIZE / SMEM_M;
					dmem_index += static_cast<std::size_t>((v_bit_len / size_of<T>::value) * BLOCK_SIZE / SMEM_M) * ld;
					cutf::cp_async::cp_async<v_bit_len>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				}
			} else if ((4 / size_of<T>::value != 0) && (ld % (4 / size_of<T>::value) == 0)) {
				constexpr unsigned v_bit_len = 4;
				const auto index = threadIdx.x * (v_bit_len / size_of<T>::value);
				const auto m = index % SMEM_M;
				const auto n = index / SMEM_M;
				auto smem_index = m + n * (SMEM_M + SKEW);
				auto dmem_index = (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;
				cutf::cp_async::cp_async<v_bit_len>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);

				for (unsigned offset = 1; offset < SMEM_M * SMEM_N / (BLOCK_SIZE * (v_bit_len/ size_of<T>::value)); offset++) {
					smem_index += (SMEM_M + SKEW) * (v_bit_len / size_of<T>::value) * BLOCK_SIZE / SMEM_M;
					dmem_index += static_cast<std::size_t>((v_bit_len / size_of<T>::value) * BLOCK_SIZE / SMEM_M) * ld;
					cutf::cp_async::cp_async<v_bit_len>(&smem_ptr[smem_index], &dmem_ptr[dmem_index]);
				}
			}
		} else {
			for (unsigned offset = 0; offset < SMEM_M * SMEM_N; offset += BLOCK_SIZE) {
				const auto index = offset + threadIdx.x;
				const auto m = index % SMEM_M;
				const auto n = index / SMEM_M;
				const auto smem_index = m + n * (SMEM_M + SKEW);
				const auto dmem_index = (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;

				T v = zero<T>();
				if ((start_m + m) < size_m && (start_n + n) < size_n) {
					v = dmem_ptr[dmem_index];
				}
				__syncwarp();
				smem_ptr[smem_index] = v;
			}
		}
	}
};

template <class _Layout, class T, unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, unsigned BLOCK_SIZE>
struct dmem_loader {
	using Layout = _Layout;
	__device__ void operator() (
			T* const smem_ptr,
			const T* const dmem_ptr,
			const unsigned ld,
			const unsigned start_m,
			const unsigned start_n,
			const unsigned size_m,
			const unsigned size_n
			) {
		dmem_loader_core<T, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE>{}(
				smem_ptr,
				dmem_ptr,
				ld,
				start_m, start_n,
				size_m, size_n
				);
	}
};

template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, unsigned BLOCK_SIZE>
struct dmem_loader<cumpsgemm::row_major, T, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE> {
	using Layout = cumpsgemm::row_major;
	__device__ void operator() (
			T* const smem_ptr,
			const T* const dmem_ptr,
			const unsigned ld,
			const unsigned start_m,
			const unsigned start_n,
			const unsigned size_m,
			const unsigned size_n
			) {
		dmem_loader_core<T, SMEM_N, SMEM_M, SKEW, BLOCK_SIZE>{}(
				smem_ptr,
				dmem_ptr,
				ld,
				start_n, start_m,
				size_n, size_m
				);
	}
};

template <unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, unsigned BLOCK_SIZE>
struct dmem_loader_conj_core {
	__device__ void operator() (
			cuComplex* const smem_ptr,
			const cuComplex* const dmem_ptr,
			const unsigned ld,
			const unsigned start_m,
			const unsigned start_n,
			const unsigned size_m,
			const unsigned size_n
			) {
		if (start_m + SMEM_M < size_m && start_n + SMEM_N < size_n) {
			for (unsigned offset = 0; offset < SMEM_M * SMEM_N; offset += BLOCK_SIZE) {
				const auto index = offset + threadIdx.x;
				const auto m = index % SMEM_M;
				const auto n = index / SMEM_M;
				const auto smem_index = m + n * (SMEM_M + SKEW);
				const auto dmem_index = (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;
				const auto v = dmem_ptr[dmem_index];
				smem_ptr[smem_index] = make_cuComplex(v.x, -v.y);
			}
		} else {
			for (unsigned offset = 0; offset < SMEM_M * SMEM_N; offset += BLOCK_SIZE) {
				const auto index = offset + threadIdx.x;
				const auto m = index % SMEM_M;
				const auto n = index / SMEM_M;
				const auto smem_index = m + n * (SMEM_M + SKEW);
				const auto dmem_index = (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;

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

template <unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, unsigned BLOCK_SIZE>
struct dmem_loader<cumpsgemm::conjugate, cuComplex, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE> {
	using Layout = cumpsgemm::row_major;
	__device__ void operator() (
			cuComplex* const smem_ptr,
			const cuComplex* const dmem_ptr,
			const unsigned ld,
			const unsigned start_m,
			const unsigned start_n,
			const unsigned size_m,
			const unsigned size_n
			) {
		dmem_loader_conj_core<SMEM_N, SMEM_M, SKEW, BLOCK_SIZE>{}(
				smem_ptr,
				dmem_ptr,
				ld,
				start_n, start_m,
				size_n, size_m
				);
	}
};

template <unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, unsigned BLOCK_SIZE>
struct dmem_loader<cumpsgemm::conjugate, float, SMEM_M, SMEM_N, SKEW, BLOCK_SIZE> {
	using Layout = cumpsgemm::col_major;
	__device__ void operator() (
			float* const,
			const float* const,
			const unsigned,
			const unsigned,
			const unsigned,
			const unsigned,
			const unsigned
			) {
		// Do nothing, only for suppressing compilation error.
	}
};

template <class T, unsigned SMEM_M, unsigned SMEM_N, unsigned SKEW, unsigned BLOCK_SIZE>
struct dmem_storer {
	__device__ void operator() (
			T* const dmem_ptr,
			const unsigned ld,
			const unsigned start_m,
			const unsigned start_n,
			const unsigned size_m,
			const unsigned size_n,
			const T* const smem_ptr,
			const T alpha, const T beta
			) {
		if (is_zero(beta)) {
			if (start_m + SMEM_M < size_m && start_n + SMEM_N < size_n) {
				const auto index = threadIdx.x;
				const auto m = index % SMEM_M;
				const auto n = index / SMEM_M;
				auto smem_index = m + n * (SMEM_M + SKEW);
				auto dmem_index = (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;
				dmem_ptr[dmem_index] = mul(smem_ptr[smem_index], alpha);

				for (unsigned offset = 1; offset < SMEM_M * SMEM_N / BLOCK_SIZE; offset++) {
					smem_index += (SMEM_M + SKEW) * (BLOCK_SIZE / SMEM_M);
					dmem_index += ld * (BLOCK_SIZE / SMEM_M);

					dmem_ptr[dmem_index] = mul(smem_ptr[smem_index], alpha);
				}
			} else {
				const auto index = threadIdx.x;
				const auto m = index % SMEM_M;
				auto n = index / SMEM_M;
				auto smem_index = m + n * (SMEM_M + SKEW);
				auto dmem_index = (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;

				for (unsigned offset = 0; offset < SMEM_M * SMEM_N; offset += BLOCK_SIZE) {
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
				for (unsigned offset = 0; offset < SMEM_M * SMEM_N; offset += BLOCK_SIZE) {
					const auto index = offset + threadIdx.x;
					const auto m = index % SMEM_M;
					const auto n = index / SMEM_M;
					const auto smem_index = m + n * (SMEM_M + SKEW);
					const auto dmem_index = (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;
					dmem_ptr[dmem_index] = mad(smem_ptr[smem_index], alpha, mul(dmem_ptr[dmem_index], beta));
				}
			} else {
				for (unsigned offset = 0; offset < SMEM_M * SMEM_N; offset += BLOCK_SIZE) {
					const auto index = offset + threadIdx.x;
					const auto m = index % SMEM_M;
					const auto n = index / SMEM_M;
					const auto smem_index = m + n * (SMEM_M + SKEW);
					const auto dmem_index = (start_m + m) + static_cast<std::size_t>(start_n + n) * ld;

					if ((start_m + m) < size_m && (start_n + n) < size_n) {
						dmem_ptr[dmem_index] = mad(smem_ptr[smem_index], alpha, mul(dmem_ptr[dmem_index], beta));
					}
					__syncwarp();
				}
			}
		}
	}
};
} // namespace device
} // namespace cumpsgemm
