#include <cstddef>
#include <cutf/device.hpp>
#include <cumpsgemm/cumpsgemm.h>
#include "handle.hpp"

#define ENABLE_A100_OPTIMAZED_PARAMETERS

extern "C" {
cublasStatus_t cuMpSGEMM_create(cuMpSGEMM_handle_t* const handle) {
	if ((*handle = new cuMpSGEMM_handle) == nullptr) {
		return CUBLAS_STATUS_INTERNAL_ERROR;
	}

	int num_sms;
	CUTF_CHECK_ERROR(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
	(*handle)->num_sms = num_sms;

	int cc_major, cc_minor;
	CUTF_CHECK_ERROR(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0));
	CUTF_CHECK_ERROR(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, 0));

	if (cc_major == 8 && cc_minor == 0) {
		cumpsgemm::configure_instance_sm80((*handle)->gemm_module, (*handle)->gemm_stridedBatch_module);
	} else {
		cumpsgemm::configure_instance_sm86((*handle)->gemm_module, (*handle)->gemm_stridedBatch_module);
	}
	//cumpsgemm::configure_instance_simt((*handle)->gemm_module, (*handle)->gemm_stridedBatch_module);

	init_exp_stats_counter_buffer((*handle));
	init_dynamic_launch_flag_buffer((*handle));

	return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cuMpSGEMM_destroy(cuMpSGEMM_handle_t handle) {
	destroy_exp_stats_counter_buffer(handle);
	destroy_launch_flag_buffer(handle);

	delete handle;
	return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cuMpSGEMM_set_stream(cuMpSGEMM_handle_t handle, const cudaStream_t cuda_stream) {
	handle->cuda_stream = cuda_stream;
	return CUBLAS_STATUS_SUCCESS;
}
}
