// ingest/ingest.h
#pragma once
#include <cstdint>
#include <vector>
#include <random>
#include "common/schema.h"

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
  // produce one tick worth of synthetic samples
  void generate(uint32_t tick_id, std::vector<SensorSample> &out);

private:
  IngestConfig cfg_;
  std::mt19937 rng_;
};