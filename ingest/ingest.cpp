// ingest/ingest.cpp
#include "ingest/ingest.h"
#include <cmath>
#include <algorithm>

Ingestor::Ingestor(const IngestConfig &cfg) : cfg_(cfg), rng_(12345) {}

void Ingestor::generate(uint32_t tick_id, std::vector<SensorSample> &out)
{
  out.clear();
  out.reserve(cfg_.junctions * cfg_.lanes_per);

  std::uniform_int_distribution<int> base(0, 10);
  std::normal_distribution<float> rush(0.f, 1.f);

  // simple diurnal pattern + noise
  float hour = std::fmod((tick_id / 3600.f), 24.f);
  float peak = (hour > 7 && hour < 9) || (hour > 16 && hour < 18) ? 1.5f : 1.0f;

  for (uint32_t j = 0; j < cfg_.junctions; ++j)
  {
    for (uint32_t l = 0; l < cfg_.lanes_per; ++l)
    {
      SensorSample s{};
      s.ts_ms = tick_id * cfg_.tick_ms;
      s.junction = (uint16_t)j;
      s.lane = (uint16_t)l;

      int b = base(rng_);
      float noise = rush(rng_) * 2.f;

      int arrivals10 = std::max(0, int((8 + b) * peak + noise) * 10);
      int q = std::max(0, int((b + (peak > 1.0f ? 5 : 1)) + std::max(0.f, noise)));
      int speed10 = std::max(1, 50 - q) * 10;

      s.arrivals = (uint16_t)arrivals10;
      s.q_len = (uint16_t)q;
      s.avg_speed = (uint16_t)speed10;
      out.push_back(s);
    }
  }
}