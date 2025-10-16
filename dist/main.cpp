// dist/main.cpp
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <thread>
#include <chrono>

#include "common/ids.h"
#include "common/schema.h"
#include "common/timers.h"
#include "ingest/ingest.h"
#include "aggregate/aggregate.h"
#include "predict/predict.h"
#include "control/control.h"

// 1s firm tick
static constexpr uint32_t TICK_MS = 1000;
static constexpr uint32_t BUDGET_P = 350;
static constexpr uint32_t BUDGET_C = 150;

static inline uint32_t env_u32(const char *n, uint32_t d)
{
  if (const char *e = std::getenv(n))
  {
    long v = std::strtol(e, nullptr, 10);
    if (v > 0 && v < 100000000)
      return (uint32_t)v;
  }
  return d;
}

static inline int stride_for_level(int level)
{
  if (level < 0)
    level = 0;
  if (level > 3)
    level = 3;
  return 1 << level; // 0→1,1→2,2→4,3→8
}

static void send_bp_to_agg(int rAgg, int level)
{
  MPI_Send(&level, 1, MPI_INT, rAgg, TAG_BP, MPI_COMM_WORLD);
}
static void drain_bp(int &accum)
{
  int flag = 0;
  MPI_Status st;
  do
  {
    MPI_Iprobe(MPI_ANY_SOURCE, TAG_BP, MPI_COMM_WORLD, &flag, &st);
    if (flag)
    {
      int lvl = 0;
      MPI_Recv(&lvl, 1, MPI_INT, st.MPI_SOURCE, TAG_BP, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      accum = std::max(accum, lvl);
    }
  } while (flag);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int world = 0, rank = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (world < 4)
  {
    std::fprintf(stderr, "FATAL: need >=4 ranks\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  const int P = std::max(1, world - 3);
  const int rCtrl = 0, rAgg = P + 1, rIng = P + 2;
  (void)rCtrl; // silence unused warning

  const uint32_t J = env_u32("JUNCTIONS", 20000);
  IngestConfig icfg{.junctions = J, .lanes_per = 3, .tick_ms = TICK_MS};
  AggConfig acfg{.junctions = J, .lanes_per = 3};
  PredConfig pcfg{.prefer_opencl = true};
  CtrlConfig ccfg{};

  if (rank == 0)
  {
    std::fprintf(stderr, "[BOOT] world=%d, predictors=%d | Ctrl=0 Agg=%d Ing=%d\n", world, P, rAgg, rIng);
    std::fflush(stderr);
  }

  const uint32_t TICKS = 40;

  if (rank == rIng)
  {
    Ingestor ing(icfg);
    std::vector<SensorSample> samples;
    uint64_t base = now_ms(), first = base + 200;
    sleep_until_ms(first);
    for (uint32_t t = 0; t < TICKS; ++t)
    {
      uint64_t tick_start = first + t * TICK_MS;
      ing.generate(t, samples);
      int cnt = (int)samples.size();
      MPI_Send(&t, 1, MPI_UNSIGNED, rAgg, TAG_FEAT, MPI_COMM_WORLD);
      MPI_Send(&cnt, 1, MPI_INT, rAgg, TAG_FEAT, MPI_COMM_WORLD);
      if (cnt > 0)
        MPI_Send(samples.data(), cnt * (int)sizeof(SensorSample), MPI_BYTE, rAgg, TAG_FEAT, MPI_COMM_WORLD);
      sleep_until_ms(tick_start + TICK_MS);
    }
  }
  else if (rank == rAgg)
  {
    Aggregator agg(acfg);
    std::vector<SensorSample> samples;
    std::vector<Features> feats, thin;
    for (uint32_t t = 0; t < TICKS; ++t)
    {
      int bp = 0;
      drain_bp(bp);
      int stride = stride_for_level(bp);

      MPI_Status st;
      uint32_t tick_id;
      int cnt = 0;
      MPI_Recv(&tick_id, 1, MPI_UNSIGNED, rIng, TAG_FEAT, MPI_COMM_WORLD, &st);
      MPI_Recv(&cnt, 1, MPI_INT, rIng, TAG_FEAT, MPI_COMM_WORLD, &st);
      samples.resize(std::max(cnt, 0));
      if (cnt > 0)
        MPI_Recv(samples.data(), cnt * (int)sizeof(SensorSample), MPI_BYTE, rIng, TAG_FEAT, MPI_COMM_WORLD, &st);

      agg.map_features(samples, feats);
      thin.clear();
      thin.reserve((feats.size() + stride - 1) / stride);
      for (size_t i = 0; i < feats.size(); i += stride)
        thin.push_back(feats[i]);

      int per = (P > 0) ? (int)thin.size() / P : 0, cursor = 0;
      for (int p = 0; p < P; ++p)
      {
        int begin = cursor, end = (p == P - 1) ? (int)thin.size() : (cursor + per);
        int n = end - begin;
        MPI_Send(&tick_id, 1, MPI_UNSIGNED, p + 1, TAG_FEAT, MPI_COMM_WORLD);
        MPI_Send(&n, 1, MPI_INT, p + 1, TAG_FEAT, MPI_COMM_WORLD);
        if (n > 0)
          MPI_Send(thin.data() + begin, n * (int)sizeof(Features), MPI_BYTE, p + 1, TAG_FEAT, MPI_COMM_WORLD);
        cursor = end;
      }
    }
  }
  else if (rank >= 1 && rank <= P)
  {
    Predictor pred(pcfg);
    std::vector<Features> feats;
    std::vector<Prediction> preds;
    for (uint32_t t = 0; t < TICKS; ++t)
    {
      MPI_Status st;
      uint32_t tick_id;
      int n = 0;
      MPI_Recv(&tick_id, 1, MPI_UNSIGNED, P + 1, TAG_FEAT, MPI_COMM_WORLD, &st);
      MPI_Recv(&n, 1, MPI_INT, P + 1, TAG_FEAT, MPI_COMM_WORLD, &st);
      feats.resize(std::max(n, 0));
      if (n > 0)
        MPI_Recv(feats.data(), n * (int)sizeof(Features), MPI_BYTE, P + 1, TAG_FEAT, MPI_COMM_WORLD, &st);

      Deadline dl{.start_ms = now_ms(), .budget_ms = BUDGET_P};
      pred.predict_batch(feats, preds);
      if (dl.elapsed() > BUDGET_P)
      {
        int level = 1;
        send_bp_to_agg(P + 1, level);
      }

      int outn = (int)preds.size();
      MPI_Send(&tick_id, 1, MPI_UNSIGNED, 0, TAG_PRED, MPI_COMM_WORLD);
      MPI_Send(&outn, 1, MPI_INT, 0, TAG_PRED, MPI_COMM_WORLD);
      if (outn > 0)
        MPI_Send(preds.data(), outn * (int)sizeof(Prediction), MPI_BYTE, 0, TAG_PRED, MPI_COMM_WORLD);
    }
  }
  else if (rank == 0)
  {
    Controller ctrl(ccfg);
    uint64_t base = now_ms(), first = base + 300;
    uint32_t misses = 0;
    for (uint32_t t = 0; t < TICKS; ++t)
    {
      uint64_t tick_start = first + t * TICK_MS;
      uint64_t tick_end = tick_start + TICK_MS;
      sleep_until_ms(tick_start);

      uint64_t t0 = now_ms();
      std::vector<Prediction> all;
      all.reserve(4096);
      int received = 0;
      while (now_ms() < tick_end && received < P)
      {
        int flag = 0;
        MPI_Status st;
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_PRED, MPI_COMM_WORLD, &flag, &st);
        if (!flag)
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          continue;
        }
        uint32_t tick_id;
        int n = 0;
        MPI_Recv(&tick_id, 1, MPI_UNSIGNED, st.MPI_SOURCE, TAG_PRED, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&n, 1, MPI_INT, st.MPI_SOURCE, TAG_PRED, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        size_t off = all.size();
        all.resize(off + std::max(n, 0));
        if (n > 0)
          MPI_Recv(all.data() + off, n * (int)sizeof(Prediction), MPI_BYTE, st.MPI_SOURCE, TAG_PRED, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        received++;
      }
      bool complete = (received == P);
      if (!complete)
        misses++;

      Deadline dctrl{.start_ms = now_ms(), .budget_ms = BUDGET_C};
      std::vector<PhaseCmd> cmds;
      ctrl.decide(all, cmds, complete);
      (void)dctrl;

      // Find the true top0 safely:
      uint32_t top0 = 9999u;
      if (!all.empty())
      {
        auto it = std::max_element(all.begin(), all.end(),
                                   [](const Prediction &a, const Prediction &b)
                                   {
                                     return a.congestion_60s < b.congestion_60s;
                                   });
        if (it != all.end())
          top0 = it->junction;
      }

      long long lat = (long long)(now_ms() - t0);
      double miss_ratio = (double)misses / (double)(t + 1);
      std::printf("[CTRL] tick %2u | slices %d/%d | preds=%zu | top0=%u | miss-ratio=%.2f | lat=%lldms\n",
                  t, complete ? P : received, P, all.size(), top0, miss_ratio, lat);
      std::fflush(stdout);

      send_bp_to_agg(P + 1, complete ? 0 : 1);
      sleep_until_ms(tick_end);
    }
  }

  MPI_Finalize();
  return 0;
}