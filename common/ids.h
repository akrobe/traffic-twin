#pragma once
#include <cstdint>

inline constexpr int TAG_FEAT = 10;
inline constexpr int TAG_PRED = 11;
inline constexpr int TAG_BP   = 12;
inline constexpr int TAG_CTRL = 13;

// roles for MPI ranks
enum class Role : int { Controller=0, Predictor=1, Aggregator=2, Ingestor=3 };