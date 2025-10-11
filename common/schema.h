#pragma once
#include <cstdint>
#include <array>

constexpr int MAX_FEATURES = 16;

struct SensorSample {
  uint32_t ts_ms;
  uint16_t junction;
  uint16_t lane;
  uint16_t q_len;      // vehicles queued
  uint16_t arrivals;   // vehicles/s *10
  uint16_t avg_speed;  // km/h *10
};

struct Features {
  uint32_t ts_ms;
  uint16_t junction;
  float f[MAX_FEATURES];
};

struct Prediction {
  uint32_t ts_ms;
  uint16_t junction;
  float congestion_60s;  // 0..1
};

struct PhaseCmd {
  uint32_t ts_ms;
  uint16_t junction;
  uint8_t  phase_id;
  uint8_t  delta_sec;
  uint8_t  reason;       // 0=MODEL,1=HEUR
};