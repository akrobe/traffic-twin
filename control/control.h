// control/control.h
#pragma once
#include <vector>
#include <cstdint>
#include "common/schema.h"

struct CtrlConfig
{
  uint8_t min_green = 8;          // not enforced in this stateless policy
  uint8_t max_green = 60;         // not enforced in this stateless policy
  uint8_t max_delta_per_tick = 6; // absolute cap for per-tick change
  // Optional: scale actions when predictions are incomplete (0..100)
  uint8_t heuristic_derate_pct = 50;
};

class Controller
{
public:
  explicit Controller(const CtrlConfig &c);
  // predictions_complete == true -> full-strength actions (MODEL)
  // predictions_complete == false -> derated actions (HEUR)
  void decide(const std::vector<Prediction> &preds,
              std::vector<PhaseCmd> &out_cmds,
              bool predictions_complete);

private:
  CtrlConfig cfg_;
};