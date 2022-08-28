#pragma once
#include <cstdint>
#include <cuComplex.h>

namespace cumpsgemm {
template <class T>
using gemm_kernel_func_t = void (*)(
			const uint64_t,
			const uint64_t,
			const uint64_t,
			const T,
			const T* const, const uint64_t,
			const T* const, const uint64_t,
			const T,
			T* const, const uint64_t
			);

template <class T>
using gemm_stridedBatch_kernel_func_t = void (*)(
			const uint64_t,
			const uint64_t,
			const uint64_t,
			const T,
			const T* const, const uint64_t, const uint64_t,
			const T* const, const uint64_t, const uint64_t,
			const T,
			T* const, const uint64_t, const uint64_t,
			const uint32_t
			);

template <class T>
struct gemm_module {
	gemm_kernel_func_t<T> kernel_func;

	unsigned smem_m, smem_n, smem_k;
};

template <class T>
struct gemm_stridedBatch_module {
	gemm_stridedBatch_kernel_func_t<T> kernel_func;

	unsigned smem_m, smem_n, smem_k;
};
} // unnamed namespace

struct cuMpSGEMM_handle {
	// 0 is for large size matmul and (num_kernel_candidates - 1) is for small.
	static constexpr unsigned num_kernel_candidates = 3;

	cumpsgemm::gemm_module<float>     sgemm_nn_func[num_kernel_candidates];
	cumpsgemm::gemm_module<float>     sgemm_tn_func[num_kernel_candidates];
	cumpsgemm::gemm_module<float>     sgemm_nt_func[num_kernel_candidates];
	cumpsgemm::gemm_module<float>     sgemm_tt_func[num_kernel_candidates];

	cumpsgemm::gemm_module<cuComplex> cgemm_nn_func[num_kernel_candidates];
	cumpsgemm::gemm_module<cuComplex> cgemm_tn_func[num_kernel_candidates];
	cumpsgemm::gemm_module<cuComplex> cgemm_cn_func[num_kernel_candidates];
	cumpsgemm::gemm_module<cuComplex> cgemm_nt_func[num_kernel_candidates];
	cumpsgemm::gemm_module<cuComplex> cgemm_tt_func[num_kernel_candidates];
	cumpsgemm::gemm_module<cuComplex> cgemm_ct_func[num_kernel_candidates];
	cumpsgemm::gemm_module<cuComplex> cgemm_nc_func[num_kernel_candidates];
	cumpsgemm::gemm_module<cuComplex> cgemm_tc_func[num_kernel_candidates];
	cumpsgemm::gemm_module<cuComplex> cgemm_cc_func[num_kernel_candidates];

	cumpsgemm::gemm_stridedBatch_kernel_func_t<float>     sgemm_stridedBatch_nn_func[num_kernel_candidates];
	cumpsgemm::gemm_stridedBatch_kernel_func_t<float>     sgemm_stridedBatch_tn_func[num_kernel_candidates];
	cumpsgemm::gemm_stridedBatch_kernel_func_t<float>     sgemm_stridedBatch_nt_func[num_kernel_candidates];
	cumpsgemm::gemm_stridedBatch_kernel_func_t<float>     sgemm_stridedBatch_tt_func[num_kernel_candidates];

	cumpsgemm::gemm_stridedBatch_kernel_func_t<cuComplex> cgemm_stridedBatch_nn_func[num_kernel_candidates];
	cumpsgemm::gemm_stridedBatch_kernel_func_t<cuComplex> cgemm_stridedBatch_tn_func[num_kernel_candidates];
	cumpsgemm::gemm_stridedBatch_kernel_func_t<cuComplex> cgemm_stridedBatch_cn_func[num_kernel_candidates];
	cumpsgemm::gemm_stridedBatch_kernel_func_t<cuComplex> cgemm_stridedBatch_nt_func[num_kernel_candidates];
	cumpsgemm::gemm_stridedBatch_kernel_func_t<cuComplex> cgemm_stridedBatch_tt_func[num_kernel_candidates];
	cumpsgemm::gemm_stridedBatch_kernel_func_t<cuComplex> cgemm_stridedBatch_ct_func[num_kernel_candidates];
	cumpsgemm::gemm_stridedBatch_kernel_func_t<cuComplex> cgemm_stridedBatch_nc_func[num_kernel_candidates];
	cumpsgemm::gemm_stridedBatch_kernel_func_t<cuComplex> cgemm_stridedBatch_tc_func[num_kernel_candidates];
	cumpsgemm::gemm_stridedBatch_kernel_func_t<cuComplex> cgemm_stridedBatch_cc_func[num_kernel_candidates];
};
