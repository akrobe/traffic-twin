// common/log.h
#pragma once
#include <cstdio>
#include <cstdint>
#include <mutex>
#include <chrono>

// Global log mutex for line integrity.
inline std::mutex &log_mutex()
{
  static std::mutex m;
  return m;
}

// Optional timestamp helper (steady ms)
inline uint64_t log_now_ms()
{
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             Clock::now().time_since_epoch())
      .count();
}

// Thread-safe single-line logger (printf-style).
template <typename... Args>
inline void LOG(const char *fmt, Args... args)
{
  std::lock_guard<std::mutex> g(log_mutex());
  // Uncomment to prefix timestamps:
  // std::fprintf(stderr, "[%llu] ", static_cast<unsigned long long>(log_now_ms()));
  std::fprintf(stderr, fmt, args...);
  std::fprintf(stderr, "\n");
  std::fflush(stderr);
}