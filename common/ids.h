// common/ids.h
#pragma once
#include <cstdint>

// Message tags (MPI)
inline constexpr int TAG_FEAT = 10; // features
inline constexpr int TAG_PRED = 11; // predictions
inline constexpr int TAG_BP = 12;   // back-pressure / control hints
inline constexpr int TAG_CTRL = 13; // control commands

// Roles for MPI ranks
enum class Role : int
{
    Controller = 0,
    Predictor = 1,
    Aggregator = 2,
    Ingestor = 3
};