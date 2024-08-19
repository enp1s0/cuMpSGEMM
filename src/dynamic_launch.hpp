#pragma once
#include "handle.hpp"
#include <cstdint>
#include <cumpsgemm/detail/common.h>

namespace cumpsgemm {
namespace dynamic_launch {
struct dynamic_launch_handle {
  unsigned flag_buffer_length;
  unsigned current_buffer_id;

  int *flag_buffer;

  bool enabled;
  unsigned enabled_id;

  cuMpSGEMM_compute_mode_t mode_A;
  cuMpSGEMM_compute_mode_t mode_B;
};

unsigned get_next_dynamic_launch_flag_buffer_id(cuMpSGEMM_handle *handle);
unsigned get_current_dynamic_launch_flag_buffer_id(cuMpSGEMM_handle *handle);
int get_dynamic_launch_buffer(cuMpSGEMM_handle *handle,
                              const unsigned buffer_id);
void set_dynamic_launch_flag_buffer_id(cuMpSGEMM_handle *handle, unsigned id);
void unset_dynamic_launch_flag_buffer_id(cuMpSGEMM_handle *handle);
} // namespace dynamic_launch
} // namespace cumpsgemm
