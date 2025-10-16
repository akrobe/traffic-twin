// common/schema.h
#pragma once
#include <cstdint>

constexpr int MAX_FEATURES = 16;

// SensorSample: compact, POD, suitable for MPI raw sends.
struct SensorSample
{
  uint32_t ts_ms; // steady-clock ms domain
  uint16_t junction;
  uint16_t lane;
  uint16_t q_len;     // vehicles queued
  uint16_t arrivals;  // vehicles/s *10
  uint16_t avg_speed; // km/h *10
};

struct Features
{
  uint32_t ts_ms; // propagated tick time
  uint16_t junction;
  float f[MAX_FEATURES]; // feature vector
};

struct Prediction
{
  uint32_t ts_ms;
  uint16_t junction;
  float congestion_60s; // 0..1
};

struct PhaseCmd
{
  uint32_t ts_ms;
  uint16_t junction;
  uint8_t phase_id;
  uint8_t delta_sec;
  uint8_t reason; // 0=MODEL,1=HEUR
};

// Basic layout sanity checks (not ABI guarantees, but catches accidental changes)
static_assert(sizeof(SensorSample) >= 12, "SensorSample unexpectedly small");
static_assert(sizeof(Features) >= 4 + 2 + sizeof(float) * MAX_FEATURES, "Features size mismatch");
static_assert(sizeof(Prediction) >= 4 + 2 + 4, "Prediction size mismatch");