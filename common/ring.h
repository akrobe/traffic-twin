#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <optional>
#include <cassert>

template<typename T>
class SpscRing {
public:
  explicit SpscRing(size_t capacity_pow2)
  : cap_(round_up_pow2(capacity_pow2)), mask_(cap_-1), buf_(cap_) {}

  bool push(const T& v) {
    auto h = head_.load(std::memory_order_relaxed);
    auto n = (h + 1) & mask_;
    if (n == tail_.load(std::memory_order_acquire)) return false; // full
    buf_[h] = v;
    head_.store(n, std::memory_order_release);
    return true;
  }

  std::optional<T> pop() {
    auto t = tail_.load(std::memory_order_relaxed);
    if (t == head_.load(std::memory_order_acquire)) return std::nullopt; // empty
    T v = buf_[t];
    tail_.store((t + 1) & mask_, std::memory_order_release);
    return v;
  }

  size_t size() const {
    auto h = head_.load(std::memory_order_acquire);
    auto t = tail_.load(std::memory_order_acquire);
    return (h - t) & mask_;
  }

  size_t capacity() const { return cap_-1; }

private:
  static size_t round_up_pow2(size_t x) {
    size_t p = 1; while (p < x) p <<= 1; return p;
  }
  const size_t cap_, mask_;
  std::vector<T> buf_;
  std::atomic<size_t> head_{0}, tail_{0};
};