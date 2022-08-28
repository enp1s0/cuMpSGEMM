#include <cumpsgemm/cumpsgemm.h>
#include "handle.hpp"

extern "C" {
cublasStatus_t cuMpSGEMM_create(cuMpSGEMM_handle_t* const handle) {
	if ((*handle = new cuMpSGEMM_handle) == nullptr) {
		return CUBLAS_STATUS_INTERNAL_ERROR;
	}

	return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cuMpSGEMM_destroy(cuMpSGEMM_handle_t handle) {
	delete handle;
	return CUBLAS_STATUS_SUCCESS;
}
}
