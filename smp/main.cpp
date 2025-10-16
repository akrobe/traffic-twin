// smp/main.cpp

#include <thread>
#include <atomic>
#include <vector>
#include <cstdio>
#include <chrono>
#include "common/ring.h"
#include "common/timers.h"
#include "ingest/ingest.h"
#include "aggregate/aggregate.h"
#include "predict/predict.h"
#include "control/control.h"

int main()
{
  IngestConfig icfg{.junctions = 2000, .lanes_per = 3, .tick_ms = 1000};
  AggConfig acfg{.junctions = icfg.junctions, .lanes_per = icfg.lanes_per};
  PredConfig pcfg{.prefer_opencl = false};
  CtrlConfig ccfg{};

  Ingestor ing(icfg);
  Aggregator agg(acfg);
  Predictor pred(pcfg);
  Controller ctrl(ccfg);

  SpscRing<std::vector<SensorSample>> ringIA(1024);
  SpscRing<std::vector<Features>> ringAP(1024);
  SpscRing<std::vector<Prediction>> ringPC(1024);

  std::atomic<bool> stop{false};
  std::atomic<uint32_t> tick{0};

  std::thread thI([&]
                  {
    while (!stop.load()) {
      std::vector<SensorSample> s; ing.generate(tick.load(), s);
      while (!ringIA.push(s)) std::this_thread::sleep_for(std::chrono::microseconds(50));
      std::this_thread::sleep_for(std::chrono::milliseconds(icfg.tick_ms));
      tick.fetch_add(1);
    } });

  std::thread thA([&]
                  {
    while (!stop.load()) {
      auto s = ringIA.pop();
      if (!s) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
      std::vector<Features> f; agg.map_features(*s, f);
      while (!ringAP.push(f)) std::this_thread::sleep_for(std::chrono::microseconds(50));
    } });

  std::thread thP([&]
                  {
    while (!stop.load()) {
      auto f = ringAP.pop();
      if (!f) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
      std::vector<Prediction> p; pred.predict_batch(*f, p);
      while (!ringPC.push(p)) std::this_thread::sleep_for(std::chrono::microseconds(50));
    } });

  std::thread thC([&]
                  {
    uint32_t printed = 0;
    while (printed < 20) {
      auto t0 = now_ms();
      auto p = ringPC.pop();
      if (!p) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
      std::vector<PhaseCmd> cmds; ctrl.decide(*p, cmds, /*complete*/true);
      auto t1 = now_ms();

      long long lat = (long long)(t1 - t0);
      std::printf("tick %u | preds=%zu | IA:%zu AP:%zu PC:%zu | lat=%lldms\n",
        printed, p->size(), ringIA.size(), ringAP.size(), ringPC.size(), lat);
      ++printed;
    }
    stop.store(true); });

  thC.join();
  thP.join();
  thA.join();
  thI.join();
  return 0;
}