// aggregate/aggregate.cpp
#include "aggregate/aggregate.h"
#include <algorithm>
#include <cmath>
#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
  constexpr float kAlpha = 0.15f;                   // EWMA
  constexpr double kTwoPi = 6.28318530717958647692; // 2*pi
  constexpr int kSecPerDay = 86400;
}

Aggregator::Aggregator(const AggConfig &c)
    : cfg_(c), ema_q_(c.junctions, 0.f) {}

void Aggregator::map_features(const std::vector<SensorSample> &samples, std::vector<Features> &out)
{
  // Guard against mismatched sample sizes
  const size_t expected = static_cast<size_t>(cfg_.junctions) * static_cast<size_t>(cfg_.lanes_per);
  assert(samples.size() == expected && "samples.size() must be junctions * lanes_per");
  if (samples.size() != expected)
  {
    out.clear();
    return;
  }

  out.assign(cfg_.junctions, Features{});

// Parallel over junctions
#pragma omp parallel for schedule(static)
  for (int j = 0; j < static_cast<int>(cfg_.junctions); ++j)
  {
    const size_t base = static_cast<size_t>(j) * cfg_.lanes_per;

    double sum_q = 0.0, sum_a = 0.0, sum_v = 0.0;
    int cnt = 0;

    for (uint32_t l = 0; l < cfg_.lanes_per; ++l)
    {
      const auto &s = samples[base + l];
      sum_q += static_cast<double>(s.q_len);
      sum_a += static_cast<double>(s.arrivals);
      sum_v += static_cast<double>(s.avg_speed);
      ++cnt;
    }

    // Defensive (should never be zero if lanes_per>0)
    if (cnt == 0)
    {
      out[j] = Features{};
      continue;
    }

    const float mq = static_cast<float>(sum_q / cnt);
    const float ma = static_cast<float>((sum_a / cnt) / 10.0); // scale to ~[0,1]
    const float mv = static_cast<float>((sum_v / cnt) / 10.0); // scale to ~[0,1]

    // EWMA of queue length (per junction)
    ema_q_[j] = kAlpha * mq + (1.f - kAlpha) * ema_q_[j];

    const uint64_t ts_ms = samples[base].ts_ms;
    const int sec = static_cast<int>((ts_ms / 1000ULL) % kSecPerDay);
    const double ang = (kTwoPi * static_cast<double>(sec)) / static_cast<double>(kSecPerDay);

    Features f{};
    f.ts_ms = ts_ms;
    f.junction = static_cast<uint16_t>(j);
    f.f[0] = mq;
    f.f[1] = ma;
    f.f[2] = mv;
    f.f[3] = ema_q_[j];
    f.f[4] = static_cast<float>(std::sin(ang)); // time-of-day sin
    f.f[5] = static_cast<float>(std::cos(ang)); // time-of-day cos
    for (int k = 6; k < MAX_FEATURES; ++k)
      f.f[k] = 0.f;

    out[j] = f;
  }
}

void Aggregator::reduce_topN(const std::vector<Features> &feats, int N, std::vector<uint16_t> &out_top, bool sort_ids)
{
  out_top.clear();
  if (N <= 0 || feats.empty())
    return;

  std::vector<std::pair<float, uint16_t>> score;
  score.reserve(feats.size());
  for (const auto &f : feats)
  {
    // Score: base queue + 0.5 * EWMA queue
    score.emplace_back(f.f[0] + 0.5f * f.f[3], f.junction);
  }

  if (N > static_cast<int>(score.size()))
    N = static_cast<int>(score.size());

  // Partition so top-N (by score desc) are in the first N slots (unordered among themselves)
  std::nth_element(
      score.begin(),
      score.begin() + N,
      score.end(),
      [](const auto &a, const auto &b)
      {
        return a.first > b.first; // higher score first
      });

  // If caller wants the *IDs sorted* deterministically, sort IDs asc.
  // Otherwise, sort the first N by score desc to present a ranked top-N list.
  if (sort_ids)
  {
    out_top.reserve(N);
    for (int i = 0; i < N; ++i)
      out_top.push_back(score[i].second);
    std::sort(out_top.begin(), out_top.end());
  }
  else
  {
    std::partial_sort(
        score.begin(),
        score.begin() + N,
        score.end(),
        [](const auto &a, const auto &b)
        { return a.first > b.first; });
    out_top.reserve(N);
    for (int i = 0; i < N; ++i)
      out_top.push_back(score[i].second);
  }
}