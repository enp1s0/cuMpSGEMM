#pragma once
#include <string>

namespace cumpsgemm {
const static std::string info_env_name = "CUMPSGEMM_INFO";
const static std::string error_env_name = "CUMPSGEMM_ERROR_LOG";
} // namespace cumpsgemm

template <class Func>
inline void cuMpSGEMM_run_if_env_defined(const std::string env_str,
                                         const Func func,
                                         const bool default_value = false) {
  const auto env = getenv(env_str.c_str());
  if ((env != nullptr && std::string(env) != "0") ||
      (env == nullptr && default_value)) {
    func();
  }
}

inline void cuMpSGEMM_log(const std::string str) {
  cuMpSGEMM_run_if_env_defined(cumpsgemm::info_env_name, [&]() {
    std::fprintf(stdout, "[cuMpSGEMM LOG] %s\n", str.c_str());
    std::fflush(stdout);
  });
}

inline void cuMpSGEMM_error(const std::string str) {
  cuMpSGEMM_run_if_env_defined(
      cumpsgemm::error_env_name,
      [&]() {
        std::fprintf(stdout, "[cuMpSGEMM ERROR] %s\n", str.c_str());
        std::fflush(stdout);
      },
      true);
}

inline void cuMpSGEMM_warning(const std::string str) {
  cuMpSGEMM_run_if_env_defined(cumpsgemm::error_env_name, [&]() {
    std::fprintf(stdout, "[cuMpSGEMM WARNING] %s\n", str.c_str());
    std::fflush(stdout);
  });
}
