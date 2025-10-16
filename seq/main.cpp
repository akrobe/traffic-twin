// seq/main.cpp

#include <vector>
#include <cstdio>
#include <algorithm>
#include "common/timers.h"
#include "ingest/ingest.h"
#include "aggregate/aggregate.h"
#include "predict/predict.h"
#include "control/control.h"

int main()
{
  IngestConfig icfg{.junctions = 20000, .lanes_per = 3, .tick_ms = 1000};
  AggConfig acfg{.junctions = icfg.junctions, .lanes_per = icfg.lanes_per};
  PredConfig pcfg{.prefer_opencl = false};
  CtrlConfig ccfg{};

  Ingestor ing(icfg);
  Aggregator agg(acfg);
  Predictor pred(pcfg);
  Controller ctrl(ccfg);

  std::vector<SensorSample> samples;
  std::vector<Features> feats;
  std::vector<Prediction> preds;
  std::vector<PhaseCmd> cmds;

  uint32_t ticks = 20;
  for (uint32_t t = 0; t < ticks; ++t)
  {
    auto t0 = now_ms();
    ing.generate(t, samples);
    auto t1 = now_ms();
    agg.map_features(samples, feats);
    auto t2 = now_ms();
    pred.predict_batch(feats, preds);
    auto t3 = now_ms();
    ctrl.decide(preds, cmds, /*complete*/ true);
    auto t4 = now_ms();
    long long lat = (long long)(t4 - t0);

    // sum-of-stages and explicit latency for Fig 9
    std::printf(
        "tick %3u | ingest %3lldms | agg %3lldms | pred %3lldms | ctrl %3lldms | lat=%lldms\n",
        t,
        (long long)(t1 - t0), (long long)(t2 - t1),
        (long long)(t3 - t2), (long long)(t4 - t3),
        lat);
  }
  return 0;
}