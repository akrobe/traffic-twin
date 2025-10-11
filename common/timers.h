#pragma once
#include <chrono>
#include <cstdint>
#include <thread>

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

inline uint64_t now_ms()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
}

// Sleep until a steady-clock millisecond timestamp (based on now_ms()).
inline void sleep_until_ms(uint64_t target_ms)
{
  uint64_t n = now_ms();
  if (target_ms > n)
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(target_ms - n));
  }
}

// Convenience timer with a fixed budget.
struct Deadline
{
  uint64_t start_ms;
  uint32_t budget_ms;
  inline uint64_t end_ms() const { return start_ms + budget_ms; }
  inline bool expired() const { return now_ms() >= end_ms(); }
  inline uint32_t elapsed() const { return (uint32_t)(now_ms() - start_ms); }
  inline uint32_t remaining() const
  {
    uint64_t n = now_ms();
    return (n >= end_ms()) ? 0u : (uint32_t)(end_ms() - n);
  }
};
