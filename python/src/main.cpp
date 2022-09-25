#include <pybind11/pybind11.h>
#include <cumpsgemm/hijack_control.hpp>

double global_lost_ratio_threshold = 0.1;

void set_compute_mode(const cuMpSGEMM_compute_mode_t compute_mode) {
	cumpsgemm::hijack_control::set_compute_mode(compute_mode);
}

void unset_compute_mode() {
	cumpsgemm::hijack_control::unset_compute_mode();
}

pybind11::list get_last_exp_stats() {
	const auto result_list = cumpsgemm::hijack_control::get_last_exp_stats();
	pybind11::list res;
	for (const auto& r : result_list) {
		pybind11::dict d;
		d["lost"] = r.first;
		d["total"]  = r.second;

		res.append(d);
	}
	return res;
}

void enable_exp_stats() {
	cumpsgemm::hijack_control::enable_exp_stats();
}

void disable_exp_stats() {
	cumpsgemm::hijack_control::disable_exp_stats();
}

void set_exp_stats_params(
		const float ignore_threshold,
		const float lost_threshold
		) {
	cumpsgemm::hijack_control::set_exp_stats_params(ignore_threshold, lost_threshold);
}

void set_global_lost_ratio_threshold(const double a) {
	global_lost_ratio_threshold = a;
}

float get_global_lost_ratio_threshold() {
	return global_lost_ratio_threshold;
}

double get_lost_ratio() {
	const auto l = cumpsgemm::hijack_control::get_last_exp_stats();
	std::size_t lost_count = 0;
	std::size_t total_count = 0;
	for (const auto es : l) {
		total_count += es.second;
		lost_count += es.first;
	}
	if (total_count > 0) {
		return static_cast<double>(lost_count) / total_count;
	}
	return 0.;
}

bool is_exp_stats_enabled() {
	return cumpsgemm::hijack_control::is_exp_stats_enabled();
}

PYBIND11_MODULE(cumpsgemm_hijack_control, m) {
	m.doc() = "cuMpSGEMM hijack control API";

	m.def("unset_compute_mode"             , &unset_compute_mode             , "unset_compute_mode");
	m.def("set_compute_mode"               , &set_compute_mode               , "set_compute_mode"  , pybind11::arg("compute_mode"));
	m.def("get_last_exp_stats"             , &get_last_exp_stats             , "get_last_exp_stats");
	m.def("enable_exp_stats"               , &enable_exp_stats               , "enable_exp_stats");
	m.def("disable_exp_stats"              , &disable_exp_stats              , "disable_exp_stats");
	m.def("set_exp_stats_params"           , &set_exp_stats_params           , "set_exp_stats_params", pybind11::arg("ignore_threshold"), pybind11::arg("lost_threshold"));
	m.def("set_global_lost_ratio_threshold", &set_global_lost_ratio_threshold, "set_global_lost_ratio_threshold", pybind11::arg("ratio_threshold"));
	m.def("get_global_lost_ratio_threshold", &get_global_lost_ratio_threshold, "get_global_lost_ratio_threshold");
	m.def("get_lost_ratio"                 , &get_lost_ratio                 , "get_lost_ratio");
	m.def("is_exp_stats_enabled"           , &is_exp_stats_enabled           , "is_exp_stats_enabled");

	pybind11::enum_<cuMpSGEMM_compute_mode_t>(m, "compute_mode")
		.value("CUMPSGEMM_CUBLAS"       , CUMPSGEMM_CUBLAS       )
		.value("CUMPSGEMM_FP16TCEC"     , CUMPSGEMM_FP16TCEC     )
		.value("CUMPSGEMM_TF32TCEC"     , CUMPSGEMM_TF32TCEC     )
		.value("CUMPSGEMM_FP16TC"       , CUMPSGEMM_FP16TC       )
		.value("CUMPSGEMM_TF32TC"       , CUMPSGEMM_TF32TC       )
		.value("CUMPSGEMM_CUBLAS_SIMT"  , CUMPSGEMM_CUBLAS_SIMT  )
		.value("CUMPSGEMM_CUBLAS_FP16TC", CUMPSGEMM_CUBLAS_FP16TC)
		.value("CUMPSGEMM_CUBLAS_TF32TC", CUMPSGEMM_CUBLAS_TF32TC)
		.export_values();
}

