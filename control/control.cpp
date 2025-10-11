#include "control/control.h"
#include <algorithm>

Controller::Controller(const CtrlConfig& c) : cfg_(c) {}

void Controller::decide(const std::vector<Prediction>& preds,
                        std::vector<PhaseCmd>& out_cmds,
                        bool predictions_complete) {
  out_cmds.clear();
  out_cmds.reserve(preds.size());
  for (const auto& p: preds) {
    uint8_t reason = predictions_complete ? 0 : 1;
    // simple guardrailed policy: more congestion => lengthen green a bit
    int delta = (int)(p.congestion_60s * cfg_.max_delta_per_tick);
    delta = std::clamp(delta, - (int)cfg_.max_delta_per_tick, (int)cfg_.max_delta_per_tick);
    uint8_t new_phase = (uint8_t)((p.junction + (delta>0?1:0)) % 4);

    out_cmds.push_back(PhaseCmd{
      p.ts_ms, p.junction, new_phase, (uint8_t)std::abs(delta), reason
    });
  }
}