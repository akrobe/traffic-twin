#pragma once
#include <vector>
#include <cstdint>
#include "common/schema.h"

struct CtrlConfig {
  uint8_t min_green = 8;
  uint8_t max_green = 60;
  uint8_t max_delta_per_tick = 6;
};

class Controller {
public:
  explicit Controller(const CtrlConfig& c);
  void decide(const std::vector<Prediction>& preds,
              std::vector<PhaseCmd>& out_cmds,
              bool predictions_complete);

private:
  CtrlConfig cfg_;
};