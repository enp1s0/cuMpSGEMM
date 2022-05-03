#include <iostream>
#include <random>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
#include <cumpsgemm/cumpsgemm.h>

constexpr unsigned min_log_N = 5;
constexpr unsigned max_log_N = 9;
constexpr unsigned log_N_interval = 2;

void test(
		cublasHandle_t const cublas_handle,
		const cublasOperation_t op_A,
		const cublasOperation_t op_B,
		const unsigned m,
		const unsigned n,
		const unsigned k,
		float* const a_ptr, const unsigned lda,
		float* const b_ptr, const unsigned ldb,
		float* const c_ptr, const unsigned ldc,
		const cuMpSGEMM_compute_mode_t compute_mode
		) {
	float alpha = 1.f, beta = 0.f;

	if (compute_mode == CUMPSGEMM_CUBLAS) {
		CUTF_CHECK_ERROR(cublasSgemm(
					cublas_handle,
					op_A, op_B,
					m, n, k,
					&alpha,
					a_ptr, lda,
					b_ptr, ldb,
					&beta,
					c_ptr, ldc
					));
	} else {
		cuMpSGEMM_sgemm(
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

	double base_norm2 = 0.;
	double diff_norm2 = 0.;

	for (unsigned i = 0; i < m; i++) {
		for (unsigned j = 0; j < n; j++) {
			double c = 0.;
			for (unsigned l = 0; l < k; l++) {
				const auto a_index = (op_A == CUBLAS_OP_N) ? (i + lda * l) : (l + lda * i);
				const auto b_index = (op_B == CUBLAS_OP_N) ? (l + ldb * j) : (j + ldb * l);

				c += static_cast<double>(a_ptr[a_index]) * static_cast<double>(b_ptr[b_index]);
			}

			const auto diff = c - c_ptr[i + j * ldc];
			base_norm2 += c * c;
			diff_norm2 += diff * diff;
		}
	}

	const auto residual = std::sqrt(diff_norm2 / base_norm2);
	std::printf("%s,%s,%s,%u,%u,%u,%e\n",
			cuMpSGEMM_get_compute_mode_string(compute_mode),
			(op_A == CUBLAS_OP_N) ? "N" : "T",
			(op_B == CUBLAS_OP_N) ? "N" : "T",
			m, n, k,
			residual
			);
}

int main() {
	float* a_ptr = cutf::memory::malloc_managed<float>((1lu << (max_log_N * 2)));
	float* b_ptr = cutf::memory::malloc_managed<float>((1lu << (max_log_N * 2)));
	float* c_ptr = cutf::memory::malloc_managed<float>((1lu << (max_log_N * 2)));

	std::mt19937 mt(0);
	std::uniform_real_distribution<float> dist(-1.f, 1.f);
	for (unsigned i = 0; i < (1lu << (max_log_N * 2)); i++) {
		a_ptr[i] = dist(mt);
		b_ptr[i] = dist(mt);
	}

	std::vector<cuMpSGEMM_compute_mode_t> modes = {
		CUMPSGEMM_CUBLAS,
		CUMPSGEMM_FP16TCEC,
		CUMPSGEMM_FP16TC,
		CUMPSGEMM_TF32TCEC,
		CUMPSGEMM_TF32TC,
	};

	auto cublas_handle_uptr = cutf::cublas::get_cublas_unique_ptr();
	for (const auto mode : modes) {
		for (unsigned log_N = min_log_N; log_N <= max_log_N; log_N += log_N_interval) {
			const auto N = 1u << log_N;
			test(
					*cublas_handle_uptr.get(),
					CUBLAS_OP_N,
					CUBLAS_OP_N,
					N, N, N,
					a_ptr, N,
					b_ptr, N,
					c_ptr, N,
					mode
					);
		}
	}
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());

	cutf::memory::free(a_ptr);
	cutf::memory::free(b_ptr);
	cutf::memory::free(c_ptr);
}
