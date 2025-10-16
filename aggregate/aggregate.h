// aggregate/aggregate.h
#pragma once
#include <vector>
#include <cstdint>
#include "common/schema.h"

struct AggConfig
{
  uint32_t junctions{0};
  uint32_t lanes_per{0};
};

class Aggregator
{
public:
  explicit Aggregator(const AggConfig &c);
  // map: compute rolling features per junction
  void map_features(const std::vector<SensorSample> &samples, std::vector<Features> &out);
  // reduce: produce top-N hotspots (junction id list)
  // If you want IDs sorted ascending (deterministic), set sort_ids=true.
  // If you want results ordered by score desc, set sort_ids=false.
  void reduce_topN(const std::vector<Features> &feats, int N, std::vector<uint16_t> &out_top, bool sort_ids = true);

private:
  AggConfig cfg_;
  std::vector<float> ema_q_; // EWMA per junction
};