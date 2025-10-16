// common/ring.h
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <optional>
#include <cassert>
#include <type_traits>

// Single-Producer Single-Consumer ring buffer.
// NOTE: Not multi-producer/consumer safe.
template <typename T>
class SpscRing
{
public:
  explicit SpscRing(size_t capacity_pow2)
      : cap_(round_up_pow2(capacity_pow2)), mask_(cap_ - 1), buf_(cap_)
  {
    // Effective capacity is cap_-1; require at least 2 usable slots
    if (cap_ < 4)
    {
      cap_ = 4;
      mask_ = cap_ - 1;
      buf_.assign(cap_, T{});
    }
  }

  bool push(const T &v)
  {
    size_t h = head_.load(std::memory_order_relaxed);
    size_t n = (h + 1) & mask_;
    if (n == tail_.load(std::memory_order_acquire))
      return false; // full
    buf_[h] = v;
    head_.store(n, std::memory_order_release);
    return true;
  }

  bool push(T &&v)
  {
    size_t h = head_.load(std::memory_order_relaxed);
    size_t n = (h + 1) & mask_;
    if (n == tail_.load(std::memory_order_acquire))
      return false; // full
    buf_[h] = std::move(v);
    head_.store(n, std::memory_order_release);
    return true;
  }

  std::optional<T> pop()
  {
    size_t t = tail_.load(std::memory_order_relaxed);
    if (t == head_.load(std::memory_order_acquire))
      return std::nullopt; // empty
    T v = std::move(buf_[t]);
    tail_.store((t + 1) & mask_, std::memory_order_release);
    return v;
  }

  [[nodiscard]] size_t size() const
  {
    size_t h = head_.load(std::memory_order_acquire);
    size_t t = tail_.load(std::memory_order_acquire);
    return (h - t) & mask_;
  }

  [[nodiscard]] bool is_empty() const
  {
    return head_.load(std::memory_order_acquire) == tail_.load(std::memory_order_acquire);
  }

  [[nodiscard]] bool is_full() const
  {
    size_t h = head_.load(std::memory_order_acquire);
    size_t n = (h + 1) & mask_;
    return n == tail_.load(std::memory_order_acquire);
  }

  [[nodiscard]] size_t capacity() const { return cap_ - 1; }

  void clear()
  {
    tail_.store(head_.load(std::memory_order_acquire), std::memory_order_release);
  }

private:
  static size_t round_up_pow2(size_t x)
  {
    size_t p = 1;
    while (p < x)
      p <<= 1;
    return p;
  }

  // Padding to mitigate false sharing between head/tail on different cores.
  alignas(64) std::atomic<size_t> head_{0};
  alignas(64) std::atomic<size_t> tail_{0};
  size_t cap_;
  size_t mask_;
  std::vector<T> buf_;
};