#pragma once
namespace cumpsgemm {
namespace dynamic_launch {
namespace utils {
__device__ __host__ inline void set_gemm_flag(
		int& flag,
		const int compute_mode
		) {
	flag = (0b1'1'00000 & flag) | compute_mode;
}

__device__ __host__ inline void set_scale_A_flag(
		int& flag,
		const bool enable_scaling 
		) {
	flag = (0b0'1'11111 & flag);
	if (enable_scaling) {
		flag |= 0b1'0'00000;
	}
}

__device__ __host__ inline void set_scale_B_flag(
		int& flag,
		const bool enable_scaling 
		) {
	flag = (0b1'0'11111 & flag);
	if (enable_scaling) {
		flag |= 0b0'1'00000;
	}
}

__device__ __host__ inline int get_gemm_flag(
		const int flag
		) {
	return flag & 0b0'0'11111;
}

__device__ __host__ inline bool get_scale_A_flag(
		const int flag
		) {
	return flag & 0b1'0'00000;
}

__device__ __host__ inline bool get_scale_B_flag(
		const int flag
		) {
	return flag & 0b0'1'00000;
}
} // detail
} // dynamic_scaling
} // cumpsgemm
