#ifndef __CUMPSGEMM_DETAIL_COMMON_H__
#define __CUMPSGEMM_DETAIL_COMMON_H__
struct cuMpSGEMM_handle;
typedef cuMpSGEMM_handle* cuMpSGEMM_handle_t;

enum cuMpSGEMM_compute_mode_t {
	CUMPSGEMM_CUBLAS           = 0,
	CUMPSGEMM_FP16TCEC         = 1,
	CUMPSGEMM_TF32TCEC         = 2,
	CUMPSGEMM_FP16TC           = 3,
	CUMPSGEMM_TF32TC           = 4,
	CUMPSGEMM_CUBLAS_SIMT      = 5,
	CUMPSGEMM_CUBLAS_FP16TC    = 6,
	CUMPSGEMM_CUBLAS_TF32TC    = 7,
	CUMPSGEMM_DRY_RUN          = 8,
	CUMPSGEMM_AUTO             = 9,
	CUMPSGEMM_UNDEFINED        = 10,
	CUMPSGEMM_FP16TCEC_SCALING = 11,
	CUMPSGEMM_FP32_SIMT        = 12,
};
#endif
