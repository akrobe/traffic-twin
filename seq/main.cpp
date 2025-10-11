#include <vector>
#include <cstdio>
#include <algorithm>
#include "common/timers.h"
#include "common/log.h"
#include "ingest/ingest.h"
#include "aggregate/aggregate.h"
#include "predict/predict.h"
#include "control/control.h"

int main(int argc, char** argv) {
  IngestConfig icfg{.junctions=500, .lanes_per=3, .tick_ms=1000};
  AggConfig    acfg{.junctions=icfg.junctions, .lanes_per=icfg.lanes_per};
  PredConfig   pcfg{.prefer_opencl=false}; // SEQ baseline: CPU only
  CtrlConfig   ccfg{};

  Ingestor ing(icfg);
  Aggregator agg(acfg);
  Predictor pred(pcfg);
  Controller ctrl(ccfg);

  std::vector<SensorSample> samples;
  std::vector<Features> feats;
  std::vector<Prediction> preds;
  std::vector<PhaseCmd> cmds;

  uint32_t ticks = 20; // short demo
  for (uint32_t t=0;t<ticks;++t) {
    auto t0 = now_ms();
    ing.generate(t, samples);
    auto t1 = now_ms();
    agg.map_features(samples, feats);
    auto t2 = now_ms();
    pred.predict_batch(feats, preds);
    auto t3 = now_ms();
    ctrl.decide(preds, cmds, /*complete*/true);
    auto t4 = now_ms();

    // top-N
    std::vector<uint16_t> top; agg.reduce_topN(feats, 10, top);

    std::printf("tick %3u | ingest %3lldms | agg %3lldms | pred %3lldms | ctrl %3lldms | top[0]=%u\n",
      t, (long long)(t1-t0),(long long)(t2-t1),(long long)(t3-t2),(long long)(t4-t3),
      top.empty()?9999:top[0]);
  }
  return 0;
}