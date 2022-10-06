#include <iostream>
#include <cutf/cublas.hpp>
#include <cutf/memory.hpp>
#include <cumpsgemm/hijack_control.hpp>

constexpr unsigned N = 10000;

int main() {
	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();
	auto mat_a = cutf::memory::get_device_unique_ptr<float>(N * N);
	auto mat_b = cutf::memory::get_device_unique_ptr<float>(N * N);
	auto mat_c = cutf::memory::get_device_unique_ptr<float>(N * N);

	float alpha = 1.0f, beta= 0.0;

	cumpsgemm::hijack_control::set_dynamic_launch_flag_buffer_id_use(0);

	cutf::cublas::gemm(
			*cublas_handle.get(),
			CUBLAS_OP_N, CUBLAS_OP_N,
			N, N, N,
			&alpha,
			mat_a.get(), N,
			mat_b.get(), N,
			&beta,
			mat_c.get(), N
			);

	cumpsgemm::hijack_control::set_dynamic_launch_flag_buffer_id_use(1);

	cutf::cublas::gemm(
			*cublas_handle.get(),
			CUBLAS_OP_N, CUBLAS_OP_N,
			N, N, N,
			&alpha,
			mat_a.get(), N,
			mat_b.get(), N,
			&beta,
			mat_c.get(), N
			);

	cudaDeviceSynchronize();
}
