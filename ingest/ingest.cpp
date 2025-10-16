// ingest/ingest.h
#pragma once
#include <cstdint>
#include <vector>
#include <random>
#include "common/schema.h"

// Synthetic sensor generator configuration.
// Units:
//   arrivals: vehicles/s scaled by 10 (uint16_t)
//   avg_speed: km/h scaled by 10 (uint16_t)
//   q_len: vehicles (uint16_t)
struct IngestConfig
{
  uint32_t junctions = 500;
  uint32_t lanes_per = 3;
  uint32_t tick_ms = 1000; // control tick
};

class Ingestor
{
public:
  explicit Ingestor(const IngestConfig &cfg);
  // Produce one tick worth of synthetic samples (deterministic seed).
  void generate(uint32_t tick_id, std::vector<SensorSample> &out);

private:
  IngestConfig cfg_;
  std::mt19937 rng_;
};