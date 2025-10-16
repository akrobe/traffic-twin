// dist/main.cpp
#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
<<<<<<< Updated upstream
=======
#include <thread>
#include <chrono>

<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
#include "common/ids.h"
#include "common/schema.h"
#include "common/timers.h"
#include "ingest/ingest.h"
#include "aggregate/aggregate.h"
#include "predict/predict.h"
#include "control/control.h"

<<<<<<< Updated upstream
<<<<<<< Updated upstream
// ------------ Real-time budgets (ms) ------------
static constexpr uint32_t TICK_MS = 1000;    // firm 1s control loop
static constexpr uint32_t BUDGET_PRED = 350; // per-slice prediction budget at P
static constexpr uint32_t BUDGET_CTRL = 150; // controller decision time
=======
=======
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes

// ------------ Back-pressure levels --------------
/*
  level 0: normal (stride=1)
  level 1: light thinning (stride=2)
  level 2: medium (stride=4)
  level 3: heavy (stride=8)
*/
static inline int stride_for_level(int level)
{
  if (level < 0)
    level = 0;
  if (level > 3)
    level = 3;
<<<<<<< Updated upstream
<<<<<<< Updated upstream
  return 1 << level;
}

static void die(const char *m)
{
  std::fprintf(stderr, "FATAL: %s\n", m);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

// Broadcast a simple BP level integer to Aggregator (rAgg)
static void send_bp_to_agg(int rAgg, int level)
{
  MPI_Send(&level, 1, MPI_INT, rAgg, TAG_BP, MPI_COMM_WORLD);
}

// Poll and drain all pending BP messages (INT) addressed to 'rank'.
static void drain_bp_for_rank(int /*rank*/, int &bp_level_accum)
=======
  return 1 << level; // 0→1,1→2,2→4,3→8
}

static void send_bp_to_agg(int rAgg, int level)
{
  MPI_Send(&level, 1, MPI_INT, rAgg, TAG_BP, MPI_COMM_WORLD);
}
static void drain_bp(int &accum)
>>>>>>> Stashed changes
=======
  return 1 << level; // 0→1,1→2,2→4,3→8
}

static void send_bp_to_agg(int rAgg, int level)
{
  MPI_Send(&level, 1, MPI_INT, rAgg, TAG_BP, MPI_COMM_WORLD);
}
static void drain_bp(int &accum)
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
<<<<<<< Updated upstream
  // Common configs
  IngestConfig icfg{.junctions = 3000, .lanes_per = 3, .tick_ms = TICK_MS};
  AggConfig acfg{.junctions = icfg.junctions, .lanes_per = icfg.lanes_per};
=======
  const uint32_t J = env_u32("JUNCTIONS", 20000);
  IngestConfig icfg{.junctions = J, .lanes_per = 3, .tick_ms = TICK_MS};
  AggConfig acfg{.junctions = J, .lanes_per = 3};
>>>>>>> Stashed changes
=======
  const uint32_t J = env_u32("JUNCTIONS", 20000);
  IngestConfig icfg{.junctions = J, .lanes_per = 3, .tick_ms = TICK_MS};
  AggConfig acfg{.junctions = J, .lanes_per = 3};
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
<<<<<<< Updated upstream
    std::vector<Features> feats;
    std::vector<Features> thin; // thinned features under BP

    int bp_level = 0; // latched per tick

    for (uint32_t t = 0; t < TOTAL_TICKS; ++t)
    {
      // Drain any BP messages before we size our work
      bp_level = 0;
      drain_bp_for_rank(rAgg, bp_level);
      int stride = stride_for_level(bp_level);

      // Receive this tick's samples from Ingestor
=======
    std::vector<Features> feats, thin;
    for (uint32_t t = 0; t < TICKS; ++t)
    {
      int bp = 0;
      drain_bp(bp);
      int stride = stride_for_level(bp);

>>>>>>> Stashed changes
=======
    std::vector<Features> feats, thin;
    for (uint32_t t = 0; t < TICKS; ++t)
    {
      int bp = 0;
      drain_bp(bp);
      int stride = stride_for_level(bp);

>>>>>>> Stashed changes
      MPI_Status st;
      uint32_t tick_id;
      int cnt = 0;
      MPI_Recv(&tick_id, 1, MPI_UNSIGNED, rIng, TAG_FEAT, MPI_COMM_WORLD, &st);
      MPI_Recv(&cnt, 1, MPI_INT, rIng, TAG_FEAT, MPI_COMM_WORLD, &st);
      samples.resize(std::max(cnt, 0));
      if (cnt > 0)
        MPI_Recv(samples.data(), cnt * (int)sizeof(SensorSample), MPI_BYTE, rIng, TAG_FEAT, MPI_COMM_WORLD, &st);

<<<<<<< Updated upstream
<<<<<<< Updated upstream
      // Map features
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
      agg.map_features(samples, feats);

      // Thin features according to BP level by stride sampling
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
<<<<<<< Updated upstream
<<<<<<< Updated upstream

    for (uint32_t t = 0; t < TOTAL_TICKS; ++t)
=======
    for (uint32_t t = 0; t < TICKS; ++t)
>>>>>>> Stashed changes
=======
    for (uint32_t t = 0; t < TICKS; ++t)
>>>>>>> Stashed changes
    {
      MPI_Status st;
      uint32_t tick_id;
      int n = 0;
      MPI_Recv(&tick_id, 1, MPI_UNSIGNED, P + 1, TAG_FEAT, MPI_COMM_WORLD, &st);
      MPI_Recv(&n, 1, MPI_INT, P + 1, TAG_FEAT, MPI_COMM_WORLD, &st);
<<<<<<< Updated upstream
<<<<<<< Updated upstream

      feats.resize(n);
=======
      feats.resize(std::max(n, 0));
>>>>>>> Stashed changes
      if (n > 0)
        MPI_Recv(feats.data(), n * (int)sizeof(Features), MPI_BYTE, P + 1, TAG_FEAT, MPI_COMM_WORLD, &st);

<<<<<<< Updated upstream
      // Prediction with per-slice budget; if exceed, ask Aggregator to thin next tick.
      Deadline dl{.start_ms = now_ms(), .budget_ms = BUDGET_PRED};
      pred.predict_batch(feats, preds);
      uint32_t dur = dl.elapsed();
      if (dur > BUDGET_PRED)
      {
        int level = 1;                                               // request light thinning
        MPI_Send(&level, 1, MPI_INT, P + 1, TAG_BP, MPI_COMM_WORLD); // to Aggregator
      }

      // Send to Controller
      MPI_Send(&tick_id, 1, MPI_UNSIGNED, 0, TAG_PRED, MPI_COMM_WORLD);
=======
      Deadline dl{.start_ms = now_ms(), .budget_ms = BUDGET_P};
      pred.predict_batch(feats, preds);
      if (dl.elapsed() > BUDGET_P)
      {
        int level = 1;
        send_bp_to_agg(P + 1, level);
      }

>>>>>>> Stashed changes
=======
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

>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
<<<<<<< Updated upstream

    // Establish a firm timeline: first tick starts a bit in the future
    uint64_t base0 = now_ms();
    uint64_t first_tick_at = base0 + 300; // 300ms from now; gives everyone time to start

    uint32_t misses = 0;

    for (uint32_t t = 0; t < TOTAL_TICKS; ++t)
=======
    uint64_t base = now_ms(), first = base + 300;
    uint32_t misses = 0;
    for (uint32_t t = 0; t < TICKS; ++t)
>>>>>>> Stashed changes
    {
      uint64_t tick_start = first + t * TICK_MS;
      uint64_t tick_end = tick_start + TICK_MS;
<<<<<<< Updated upstream

      // Align to tick start (optional; keeps cadence clean)
=======
>>>>>>> Stashed changes
=======
    uint64_t base = now_ms(), first = base + 300;
    uint32_t misses = 0;
    for (uint32_t t = 0; t < TICKS; ++t)
    {
      uint64_t tick_start = first + t * TICK_MS;
      uint64_t tick_end = tick_start + TICK_MS;
>>>>>>> Stashed changes
      sleep_until_ms(tick_start);

      uint64_t t0 = now_ms();
      std::vector<Prediction> all;
      all.reserve(4096);
<<<<<<< Updated upstream
<<<<<<< Updated upstream

      int received_from = 0;
      // Non-blocking gather loop until deadline or all predictors reported
      while (now_ms() < tick_end && received_from < P)
=======
      int received = 0;
      while (now_ms() < tick_end && received < P)
>>>>>>> Stashed changes
=======
      int received = 0;
      while (now_ms() < tick_end && received < P)
>>>>>>> Stashed changes
      {
        int flag = 0;
        MPI_Status st;
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_PRED, MPI_COMM_WORLD, &flag, &st);
        if (!flag)
        {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          continue;
        }
        // Receive one slice
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

<<<<<<< Updated upstream
<<<<<<< Updated upstream
      // Controller decision (budgeted)
      Deadline dctrl{.start_ms = now_ms(), .budget_ms = BUDGET_CTRL};
      std::vector<PhaseCmd> cmds;
      ctrl.decide(all, cmds, /*predictions_complete*/ complete);
      (void)dctrl; // reserved for future accounting

      // Quick visibility: top1 and running miss ratio
=======
=======
>>>>>>> Stashed changes
      Deadline dctrl{.start_ms = now_ms(), .budget_ms = BUDGET_C};
      std::vector<PhaseCmd> cmds;
      ctrl.decide(all, cmds, complete);
      (void)dctrl;

      // Find the true top0 safely:
      uint32_t top0 = 9999u;
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream

      long long lat = (long long)(now_ms() - t0);
      double miss_ratio = (double)misses / (double)(t + 1);
      std::printf("[CTRL] tick %2u | slices %d/%d | preds=%zu | top0=%u | miss-ratio=%.2f | lat=%lldms\n",
                  t, complete ? P : received, P, all.size(), top0, miss_ratio, lat);
      std::fflush(stdout);

<<<<<<< Updated upstream
      // If we missed the deadline, request stronger back-pressure for next tick.
      if (!complete)
      {
        int level = (miss_ratio > 0.20) ? 3 : (miss_ratio > 0.10 ? 2 : 1);
        send_bp_to_agg(P + 1, level); // to Aggregator
      }
      else
      {
        // When healthy, nudge Aggregator back toward normal
        int level = 0;
        send_bp_to_agg(P + 1, level);
      }

      // Hold the 1s cadence
=======
      send_bp_to_agg(P + 1, complete ? 0 : 1);
>>>>>>> Stashed changes
=======

      long long lat = (long long)(now_ms() - t0);
      double miss_ratio = (double)misses / (double)(t + 1);
      std::printf("[CTRL] tick %2u | slices %d/%d | preds=%zu | top0=%u | miss-ratio=%.2f | lat=%lldms\n",
                  t, complete ? P : received, P, all.size(), top0, miss_ratio, lat);
      std::fflush(stdout);

      send_bp_to_agg(P + 1, complete ? 0 : 1);
>>>>>>> Stashed changes
      sleep_until_ms(tick_end);
    }
  }

  MPI_Finalize();
  return 0;
}