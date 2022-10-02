#include <pybind11/pybind11.h>
#include <cumpsgemm/hijack_control.hpp>

void set_compute_mode(const cuMpSGEMM_compute_mode_t compute_mode) {
	cumpsgemm::hijack_control::set_compute_mode(compute_mode);
}

void unset_compute_mode() {
	cumpsgemm::hijack_control::unset_compute_mode();
}

PYBIND11_MODULE(cumpsgemm_hijack_control, m) {
	m.doc() = "cuMpSGEMM hijack control API";

	m.def("unset_compute_mode", &unset_compute_mode, "unset_compute_mode");
	m.def("set_compute_mode"  , &set_compute_mode  , "set_compute_mode"  , pybind11::arg("compute_mode"));

	pybind11::enum_<cuMpSGEMM_compute_mode_t>(m, "compute_mode")
		.value("CUMPSGEMM_CUBLAS"       , CUMPSGEMM_CUBLAS       )
		.value("CUMPSGEMM_FP16TCEC"     , CUMPSGEMM_FP16TCEC     )
		.value("CUMPSGEMM_TF32TCEC"     , CUMPSGEMM_TF32TCEC     )
		.value("CUMPSGEMM_FP16TC"       , CUMPSGEMM_FP16TC       )
		.value("CUMPSGEMM_TF32TC"       , CUMPSGEMM_TF32TC       )
		.value("CUMPSGEMM_CUBLAS_SIMT"  , CUMPSGEMM_CUBLAS_SIMT  )
		.value("CUMPSGEMM_CUBLAS_FP16TC", CUMPSGEMM_CUBLAS_FP16TC)
		.value("CUMPSGEMM_CUBLAS_TF32TC", CUMPSGEMM_CUBLAS_TF32TC)
		.value("CUMPSGEMM_DRY_RUN"      , CUMPSGEMM_DRY_RUN      )
		.export_values();
}

