#pragma once
#include <vector>
#include "common/schema.h"

struct AggConfig {
  uint32_t junctions;
  uint32_t lanes_per;
};

class Aggregator {
public:
  explicit Aggregator(const AggConfig& c);
  // map: compute rolling features per junction
  void map_features(const std::vector<SensorSample>& samples, std::vector<Features>& out);
  // reduce: produce top-N hotspots (junction id list)
  void reduce_topN(const std::vector<Features>& feats, int N, std::vector<uint16_t>& out_top);

private:
  AggConfig cfg_;
  std::vector<float> ema_q_; // EWMA per junction
};