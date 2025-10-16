// common/timers.h
#pragma once
#include <chrono>
#include <cstdint>
#include <thread>

using Clock = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;

[[nodiscard]] inline uint64_t now_ms()
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             Clock::now().time_since_epoch())
      .count();
}

// Sleep until a steady-clock millisecond timestamp (same domain as now_ms()).
// Uses a short spin+yield near the target to reduce oversleep jitter on macOS.
inline void sleep_until_ms(uint64_t target_ms)
{
  for (;;)
  {
    uint64_t n = now_ms();
    if (n >= target_ms)
      break;
    const uint64_t remain = target_ms - n;
    if (remain > 2)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(remain - 1));
    }
    else
    {
      // final 0–2ms: yield/spin to avoid overshoot
      std::this_thread::yield();
    }
  }
}

// Convenience timer with a fixed budget.
struct Deadline
{
  uint64_t start_ms{now_ms()};
  uint32_t budget_ms{1000};

  [[nodiscard]] inline uint64_t end_ms() const { return start_ms + budget_ms; }
  [[nodiscard]] inline bool expired() const { return now_ms() >= end_ms(); }
  [[nodiscard]] inline uint32_t elapsed() const { return static_cast<uint32_t>(now_ms() - start_ms); }
  [[nodiscard]] inline uint32_t remaining() const
  {
    uint64_t n = now_ms();
    return (n >= end_ms()) ? 0u : static_cast<uint32_t>(end_ms() - n);
  }
};