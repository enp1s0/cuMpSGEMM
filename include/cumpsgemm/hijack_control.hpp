#pragma once
#include <vector>
#include "detail/common.h"

namespace cumpsgemm {
namespace hijack_control {
cuMpSGEMM_handle_t get_internal_global_handle();

void set_compute_mode(const cuMpSGEMM_compute_mode_t mode);
void unset_compute_mode();

void enable_exp_stats();
void disable_exp_stats();
std::vector<std::pair<std::size_t, std::size_t>> get_last_exp_stats();
} // namespace hijack_control
} // namespace cumpsgemm
