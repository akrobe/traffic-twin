// control/control.cpp
#include "control/control.h"
#include <algorithm>
#include <cmath>

namespace
{
  // Map congestion [0,1] -> signed delta in [-max_delta, max_delta]
  inline int congestion_to_delta(float c01, int max_delta)
  {
    // Linear policy: >=0.5 tends to lengthen green; <0.5 shortens/stays
    // You can tweak bias/curve later if you want.
    float scaled = std::clamp(c01, 0.0f, 1.0f) * static_cast<float>(max_delta);
    return static_cast<int>(std::lround(scaled));
  }

  // Safe u8 of absolute value clamped to max
  inline uint8_t clamp_abs_u8(int v, int max_v)
  {
    int a = std::min(std::abs(v), std::max(0, max_v));
    return static_cast<uint8_t>(a);
  }

  // Simple 4-phase ring
  inline uint8_t next_phase_for_delta(uint16_t junction, int delta)
  {
    // If we’re “lengthening”, bias to next phase; otherwise keep current mapping
    uint8_t phase = static_cast<uint8_t>(junction % 4);
    if (delta > 0)
      phase = static_cast<uint8_t>((phase + 1) % 4);
    return phase;
  }
} // namespace

Controller::Controller(const CtrlConfig &c) : cfg_(c) {}

void Controller::decide(const std::vector<Prediction> &preds,
                        std::vector<PhaseCmd> &out_cmds,
                        bool predictions_complete)
{
  out_cmds.clear();
  out_cmds.reserve(preds.size());

  // Heuristic de-rate when predictions are incomplete (noise/partial coverage)
  const int derate_pct = predictions_complete ? 100 : std::clamp<int>(cfg_.heuristic_derate_pct, 0, 100);
  const uint8_t reason = predictions_complete ? 0 /*MODEL*/ : 1 /*HEUR*/;

  for (const auto &p : preds)
  {
    // 1) compute signed delta
    int raw = congestion_to_delta(p.congestion_60s, cfg_.max_delta_per_tick);
    // 2) apply derate if incomplete
    raw = (raw * derate_pct) / 100;
    // 3) clamp to policy envelope
    raw = std::clamp(raw, -static_cast<int>(cfg_.max_delta_per_tick),
                     static_cast<int>(cfg_.max_delta_per_tick));
    // 4) select a phase in a simple 4-phase ring
    uint8_t new_phase = next_phase_for_delta(p.junction, raw);
    // 5) encode absolute delta seconds (safely bounded to u8)
    uint8_t delta_sec = clamp_abs_u8(raw, cfg_.max_delta_per_tick);

    out_cmds.push_back(PhaseCmd{
        p.ts_ms,
        p.junction,
        new_phase,
        delta_sec,
        reason});
  }

  // NOTE: We deliberately do not sort cmds so output order matches preds input.
  // If you prefer top-heavy logs, you could sort by highest congestion here.
}