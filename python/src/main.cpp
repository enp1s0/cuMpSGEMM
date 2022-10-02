#include <utility>
#include <pybind11/pybind11.h>
#include <cumpsgemm/hijack_control.hpp>

double global_lost_ratio_threshold = 0.1;
int global_auto_kernel_selection_enabled = 0;
unsigned global_cublas_dim_mn_threshold = 128;
unsigned global_cublas_dim_k_threshold = 64;

namespace cumpsgemm {
namespace hijack_control {
void set_compute_mode(const cuMpSGEMM_compute_mode_t) {};
void unset_compute_mode() {};
std::pair<std::size_t, std::size_t> get_exp_stats(const unsigned buffer_id) {return std::pair<std::size_t, std::size_t>{1, 1};};
void enable_exp_stats() {};
void disable_exp_stats() {};
void set_exp_stats_params(const float, const float) {};
bool is_exp_stats_enabled(){return false;};
unsigned get_current_buffer_id() {return 0;}
void exp_stats(const unsigned m, const unsigned n, const float* const ptr, const unsigned ld, const unsigned batch_size, const unsigned stride){};
}
}

void set_compute_mode(const cuMpSGEMM_compute_mode_t compute_mode) {
	cumpsgemm::hijack_control::set_compute_mode(compute_mode);
}

void unset_compute_mode() {
	cumpsgemm::hijack_control::unset_compute_mode();
}

pybind11::dict get_exp_stats(const unsigned buffer_id) {
	const auto r = cumpsgemm::hijack_control::get_exp_stats(buffer_id);
	pybind11::dict d;
	d["lost"] = r.first;
	d["total"]  = r.second;
	return d;
}

void enable_exp_stats() {
	cumpsgemm::hijack_control::enable_exp_stats();
}

void disable_exp_stats() {
	cumpsgemm::hijack_control::disable_exp_stats();
}

void enable_auto_kernel_selection() {
	global_auto_kernel_selection_enabled = 1;
}

void disable_auto_kernel_selection() {
	global_auto_kernel_selection_enabled = 0;
}

bool is_auto_kernel_selection_enabled() {
	return global_auto_kernel_selection_enabled;
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

double get_lost_ratio(const unsigned buffer_id) {
	const auto r = cumpsgemm::hijack_control::get_exp_stats(buffer_id);
	std::size_t lost_count = r.first;
	std::size_t total_count = r.second;
	if (total_count > 0) {
		return static_cast<double>(lost_count) / total_count;
	}
	return 0.;
}

bool is_exp_stats_enabled() {
	return cumpsgemm::hijack_control::is_exp_stats_enabled();
}

unsigned get_current_buffer_id() {
	return cumpsgemm::hijack_control::get_current_buffer_id();
}

unsigned get_global_cublas_dim_mn_threshold() {
	return global_cublas_dim_mn_threshold;
}

void set_global_cublas_dim_mn_threshold(const unsigned dim) {
	global_cublas_dim_mn_threshold = dim;
}

unsigned get_global_cublas_dim_k_threshold() {
	return global_cublas_dim_k_threshold;
}

void set_global_cublas_dim_k_threshold(const unsigned dim) {
	global_cublas_dim_k_threshold = dim;
}

void exp_stats(
	const unsigned m,
	const unsigned n,
	const long ptr,
	const unsigned ld,
	const unsigned batch_size,
	const unsigned stride
	) {
	cumpsgemm::hijack_control::exp_stats(
		m, n,
		reinterpret_cast<float*>(ptr), ld,
		batch_size,
		stride
		);
}

PYBIND11_MODULE(cumpsgemm_hijack_control, m) {
	m.doc() = "cuMpSGEMM hijack control API";

	m.def("unset_compute_mode"                 , &unset_compute_mode                 , "unset_compute_mode");
	m.def("set_compute_mode"                   , &set_compute_mode                   , "set_compute_mode"  , pybind11::arg("compute_mode"));
	m.def("get_exp_stats"                      , &get_exp_stats                      , "get_exp_stats", pybind11::arg("buffer_id"));
	m.def("get_current_buffer_id"              , &get_current_buffer_id              , "get_current_buffer_id");
	m.def("enable_exp_stats"                   , &enable_exp_stats                   , "enable_exp_stats");
	m.def("disable_exp_stats"                  , &disable_exp_stats                  , "disable_exp_stats");
	m.def("set_exp_stats_params"               , &set_exp_stats_params               , "set_exp_stats_params", pybind11::arg("ignore_threshold"), pybind11::arg("lost_threshold"));
	m.def("set_global_lost_ratio_threshold"    , &set_global_lost_ratio_threshold    , "set_global_lost_ratio_threshold", pybind11::arg("ratio_threshold"));
	m.def("get_global_lost_ratio_threshold"    , &get_global_lost_ratio_threshold    , "get_global_lost_ratio_threshold");
	m.def("get_lost_ratio"                     , &get_lost_ratio                     , "get_lost_ratio", pybind11::arg("buffer_id"));
	m.def("is_exp_stats_enabled"               , &is_exp_stats_enabled               , "is_exp_stats_enabled");
	m.def("enable_auto_kernel_selection"       , &enable_auto_kernel_selection       , "enable_auto_kernel_selection");
	m.def("disable_auto_kernel_selection"      , &disable_auto_kernel_selection      , "disable_auto_kernel_selection");
	m.def("is_auto_kernel_selection_enabled"   , &is_auto_kernel_selection_enabled   , "is_auto_kernel_selection_enabled");
	m.def("set_global_cublas_dim_mn_threshold" , &set_global_cublas_dim_mn_threshold , "set_global_cublas_dim_mn_threshold", pybind11::arg("dim"));
	m.def("get_global_cublas_dim_mn_threshold" , &get_global_cublas_dim_mn_threshold , "get_global_cublas_dim_mn_threshold");
	m.def("set_global_cublas_dim_k_threshold"  , &set_global_cublas_dim_k_threshold  , "set_global_cublas_dim_k_threshold", pybind11::arg("dim"));
	m.def("get_global_cublas_dim_k_threshold"  , &get_global_cublas_dim_k_threshold  , "get_global_cublas_dim_k_threshold");
	m.def("exp_stats"                          , &exp_stats                          , "exp_stats", pybind11::arg("m"), pybind11::arg("n"), pybind11::arg("ptr"), pybind11::arg("ld"), pybind11::arg("batch_size") = 1, pybind11::arg("stride") = 0);



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

