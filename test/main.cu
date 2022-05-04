#include <iostream>
#include <vector>
#include <cutf/curand.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include <cumpsgemm/cumpsgemm.hpp>

constexpr unsigned min_log_N = 5;
constexpr unsigned max_log_N = 11;
constexpr unsigned log_N_interval = 2;

double error_threshold(
		const cuMpSGEMM_compute_mode_t compute_mode,
		const std::size_t N
		) {
	if (compute_mode == CUMPSGEMM_FP16TC ||
			compute_mode == CUMPSGEMM_TF32TC) {
		return 1. / (1 << 10) * std::sqrt(N);
	}
	return 1. / (1 << 23) * std::sqrt(N);
}

__device__ double mad(
		const float a,
		const float b,
		const double c
		) {
	return static_cast<double>(a) * static_cast<double>(b) + c;
}

__device__ double2 mad(
		const float2 a,
		const float2 b,
		const double2 c
		) {
	const auto dbl_a = cuComplexFloatToDouble(a);
	const auto dbl_b = cuComplexFloatToDouble(b);
	return cuCadd(cuCmul(dbl_a, dbl_b), c);
}

template <class T>
struct doubled_t {using type = double;};
template <> struct doubled_t<cuComplex> {using type = cuDoubleComplex;};

template <class T>
__device__ T load_with_op(
		const T* const ptr,
		cublasOperation_t op
		) {
	return *ptr;
}

template <>
__device__ cuComplex load_with_op<cuComplex>(
		const cuComplex* const ptr,
		cublasOperation_t op
		) {
	if (op == CUBLAS_OP_C) {
		const auto v = *ptr;
		return cuConjf(v);
	}
	return *ptr;
}

__device__ double diff2(
		const cuDoubleComplex ab,
		const cuComplex c
		) {
	const auto real_diff = ab.x - c.x;
	const auto imag_diff = ab.y - c.y;
	return real_diff * real_diff + imag_diff * imag_diff;
}
__device__ double diff2(
		const double ab,
		const float c
		) {
	const auto diff = ab - c;
	return diff * diff;
}
__device__ double norm2(
		const cuDoubleComplex a
		) {
	return a.x * a.x + a.y * a.y;
}
__device__ double norm2(
		const double a
		) {
	return a * a;
}


template <class T>
__host__ __device__ T one() {return 1;}
template <> __host__ __device__ cuComplex one() {return make_cuComplex(1, 0);}
template <class T>
__host__ __device__ T zero() {return 0;}
template <> __host__ __device__ cuComplex zero() {return make_cuComplex(0, 0);}
template <> __host__ __device__ cuDoubleComplex zero() {return make_cuDoubleComplex(0, 0);}


template <class T>
__global__ void calc_matmul_residual_kernel(
		double* const base_norm2_ptr,
		double* const diff_norm2_ptr,
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const unsigned m,
		const unsigned n,
		const unsigned k,
		const T* const a_ptr, const unsigned lda,
		const T* const b_ptr, const unsigned ldb,
		const T* const c_ptr, const unsigned ldc
		) {
	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= m * n) return;

	const auto c_m = tid % m;
	const auto c_n = tid / m;

	auto c = zero<typename doubled_t<T>::type>();
	for (std::size_t ik = 0; ik < k; ik++) {
		std::size_t a_index = 0;
		if (op_A == CUBLAS_OP_C || op_A == CUBLAS_OP_N) {
			a_index = c_m + ik * lda;
		} else {
			a_index = ik + c_m * lda;
		}

		std::size_t b_index = 0;
		if (op_B == CUBLAS_OP_C || op_B == CUBLAS_OP_N) {
			b_index = ik + c_n * ldb;
		} else {
			b_index = c_n + ik * ldb;
		}

		c = mad(
				load_with_op(a_ptr + a_index, op_A),
				load_with_op(b_ptr + b_index, op_B),
				c
				);
		const auto aa = load_with_op(a_ptr + a_index, op_A);
		const auto bb = load_with_op(b_ptr + b_index, op_B);
	}
	const auto base_norm2 = norm2(c);
	const auto diff_norm2 = diff2(c, c_ptr[c_m + c_n * ldc]);

	atomicAdd(base_norm2_ptr, base_norm2);
	atomicAdd(diff_norm2_ptr, diff_norm2);
}

template <class T>
double calc_matmul_residual(
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const unsigned m,
		const unsigned n,
		const unsigned k,
		const T* const a_ptr, const unsigned lda,
		const T* const b_ptr, const unsigned ldb,
		const T* const c_ptr, const unsigned ldc
		) {
	auto base_norm2_ptr = cutf::memory::malloc_managed<double>(1);
	auto diff_norm2_ptr = cutf::memory::malloc_managed<double>(1);

	*base_norm2_ptr = 0;
	*diff_norm2_ptr = 0;

	constexpr unsigned block_size = 256;
	const auto num_threads = m * n;
	const auto grid_size = (num_threads + block_size - 1) / block_size;

	cudaDeviceSynchronize();
	calc_matmul_residual_kernel<<<grid_size, block_size>>>(
			base_norm2_ptr, diff_norm2_ptr,
			op_A, op_B,
			m, n, k,
			a_ptr, lda,
			b_ptr, ldb,
			c_ptr, ldc
			);
	cudaDeviceSynchronize();

	const auto residual = std::sqrt(*diff_norm2_ptr / *base_norm2_ptr);

	cutf::memory::free(base_norm2_ptr);
	cutf::memory::free(diff_norm2_ptr);

	return residual;
}

void cublas_gemm(
		cublasHandle_t const cublas_handle,
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const unsigned m,
		const unsigned n,
		const unsigned k,
		const float* const alpha,
		const float* const a_ptr, const unsigned lda,
		const float* const b_ptr, const unsigned ldb,
		const float* const beta,
		float* const c_ptr, const unsigned ldc
		) {
		CUTF_CHECK_ERROR(cublasSgemm(
					cublas_handle,
					op_A, op_B,
					m, n, k,
					alpha,
					a_ptr, lda,
					b_ptr, ldb,
					beta,
					c_ptr, ldc
					));
}

void cublas_gemm(
		cublasHandle_t const cublas_handle,
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const unsigned m,
		const unsigned n,
		const unsigned k,
		const cuComplex* const alpha,
		const cuComplex* const a_ptr, const unsigned lda,
		const cuComplex* const b_ptr, const unsigned ldb,
		const cuComplex* const beta,
		cuComplex* const c_ptr, const unsigned ldc
		) {
		CUTF_CHECK_ERROR(cublasCgemm(
					cublas_handle,
					op_A, op_B,
					m, n, k,
					alpha,
					a_ptr, lda,
					b_ptr, ldb,
					beta,
					c_ptr, ldc
					));
}

template <class T>
void sgemm_test_core(
		cublasHandle_t const cublas_handle,
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const unsigned m,
		const unsigned n,
		const unsigned k,
		T* const a_ptr, const unsigned lda,
		T* const b_ptr, const unsigned ldb,
		T* const c_ptr, const unsigned ldc,
		const cuMpSGEMM_compute_mode_t compute_mode
		) {
	const auto alpha = one<T>(), beta = zero<T>();

	if (compute_mode == CUMPSGEMM_CUBLAS) {
		cublas_gemm(
				cublas_handle,
				op_A, op_B,
				m, n, k,
				&alpha,
				a_ptr, lda,
				b_ptr, ldb,
				&beta,
				c_ptr, ldc
				);
	} else {
		cumpsgemm::gemm(
				op_A, op_B,
				m, n, k,
				&alpha,
				a_ptr, lda,
				b_ptr, ldb,
				&beta,
				c_ptr, ldc,
				compute_mode
				);
	}
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());

	const auto residual = calc_matmul_residual(
					op_A, op_B,
					m, n, k,
					a_ptr, lda,
					b_ptr, ldb,
					c_ptr, ldc
			);
	std::printf("%s,%s,%s,%s,%u,%u,%u,%e,%s\n",
			(std::is_same<float, T>::value ? "sgemm" : "cgemm"),
			cuMpSGEMM_get_compute_mode_string(compute_mode),
			(op_A == CUBLAS_OP_N) ? "N" : ((op_A == CUBLAS_OP_T) ? "T" : "C"),
			(op_B == CUBLAS_OP_N) ? "N" : ((op_B == CUBLAS_OP_T) ? "T" : "C"),
			m, n, k,
			residual,
			(residual < error_threshold(compute_mode, m) ? "OK" : "NG")
			);
	std::fflush(stdout);
}

int main() {
	constexpr uint64_t seed = 0;
	constexpr std::size_t max_num_elements = (1lu << (max_log_N * 2)) * 2;
	float* a_ptr = cutf::memory::malloc<float>(max_num_elements);
	float* b_ptr = cutf::memory::malloc<float>(max_num_elements);
	float* c_ptr = cutf::memory::malloc<float>(max_num_elements);

	auto curand_gen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_PHILOX4_32_10);
	CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*curand_gen.get(), seed));
	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*curand_gen.get(), a_ptr, max_num_elements));
	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*curand_gen.get(), b_ptr, max_num_elements));

	std::vector<cuMpSGEMM_compute_mode_t> modes = {
		CUMPSGEMM_CUBLAS,
		CUMPSGEMM_FP16TCEC,
		CUMPSGEMM_FP16TC,
		CUMPSGEMM_TF32TCEC,
		CUMPSGEMM_TF32TC,
	};
	std::vector<cublasOperation_t> sgemm_ops = {
		CUBLAS_OP_N,
		CUBLAS_OP_T
	};
	std::vector<cublasOperation_t> cgemm_ops = {
		CUBLAS_OP_N,
		CUBLAS_OP_T,
		CUBLAS_OP_C
	};

	auto cublas_handle_uptr = cutf::cublas::get_cublas_unique_ptr();
	for (const auto mode : modes) {
		for (const auto op_A : sgemm_ops) {
			for (const auto op_B : sgemm_ops) {
				for (unsigned log_N = min_log_N; log_N <= max_log_N; log_N += log_N_interval) {
					const auto N = 1u << log_N;
					sgemm_test_core(
							*cublas_handle_uptr.get(),
							op_A,
							op_B,
							N, N, N,
							a_ptr, N,
							b_ptr, N,
							c_ptr, N,
							mode
							);
				}
			}
		}
	}
	for (const auto mode : modes) {
		for (const auto op_A : cgemm_ops) {
			for (const auto op_B : cgemm_ops) {
				for (unsigned log_N = min_log_N; log_N <= max_log_N; log_N += log_N_interval) {
					const auto N = 1u << log_N;
					sgemm_test_core(
							*cublas_handle_uptr.get(),
							op_A,
							op_B,
							N, N, N,
							reinterpret_cast<cuComplex*>(a_ptr), N,
							reinterpret_cast<cuComplex*>(b_ptr), N,
							reinterpret_cast<cuComplex*>(c_ptr), N,
							mode
							);
				}
			}
		}
	}
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());

	cutf::memory::free(a_ptr);
	cutf::memory::free(b_ptr);
	cutf::memory::free(c_ptr);
}
