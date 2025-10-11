#pragma once
#include <cstdio>
#include <cstdint>
#include <mutex>

inline std::mutex& log_mutex() { static std::mutex m; return m; }

template<typename... Args>
inline void LOG(const char* fmt, Args... args) {
  std::lock_guard<std::mutex> g(log_mutex());
  std::fprintf(stderr, fmt, args...);
  std::fprintf(stderr, "\n");
}