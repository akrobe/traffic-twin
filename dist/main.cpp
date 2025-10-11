#include <mpi.h>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include "common/ids.h"
#include "common/schema.h"
#include "common/timers.h"
#include "ingest/ingest.h"
#include "aggregate/aggregate.h"
#include "predict/predict.h"
#include "control/control.h"

// ------------ Real-time budgets (ms) ------------
static constexpr uint32_t TICK_MS = 1000;    // firm 1s control loop
static constexpr uint32_t BUDGET_PRED = 350; // per-slice prediction budget at P
static constexpr uint32_t BUDGET_CTRL = 150; // controller decision time

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
      bp_level_accum = std::max(bp_level_accum, lvl); // keep strongest
    }
  } while (flag);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int P = std::max(1, world - 3); // predictors
  int rCtrl = 0, rAgg = P + 1, rIng = P + 2;
  if (world < 4)
    die("need at least 4 ranks: C, P, A, I");

  // Common configs
  IngestConfig icfg{.junctions = 3000, .lanes_per = 3, .tick_ms = TICK_MS};
  AggConfig acfg{.junctions = icfg.junctions, .lanes_per = icfg.lanes_per};
  PredConfig pcfg{.prefer_opencl = true};
  CtrlConfig ccfg{};

  const uint32_t TOTAL_TICKS = 40;

  // ---------------- Ingestor ----------------
  if (rank == rIng)
  {
    Ingestor ing(icfg);
    std::vector<SensorSample> samples;

    // Align to next second for a clean start
    uint64_t base0 = now_ms();
    uint64_t first_tick_at = base0 + 200; // 200ms from now
    sleep_until_ms(first_tick_at);

    for (uint32_t t = 0; t < TOTAL_TICKS; ++t)
    {
      uint64_t tick_start = first_tick_at + t * TICK_MS;

      // Generate synchronously at tick start
      ing.generate(t, samples);

      // Send to Aggregator: tick id + count + payload (bytes)
      int cnt = (int)samples.size();
      MPI_Send(&t, 1, MPI_UNSIGNED, rAgg, TAG_FEAT, MPI_COMM_WORLD);
      MPI_Send(&cnt, 1, MPI_INT, rAgg, TAG_FEAT, MPI_COMM_WORLD);
      MPI_Send(samples.data(), cnt * sizeof(SensorSample), MPI_BYTE, rAgg, TAG_FEAT, MPI_COMM_WORLD);

      // Sleep until next tick boundary
      sleep_until_ms(tick_start + TICK_MS);
    }
  }

  // ---------------- Aggregator ----------------
  else if (rank == rAgg)
  {
    Aggregator agg(acfg);
    std::vector<SensorSample> samples;
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
      MPI_Status st;
      uint32_t tick_id;
      int cnt = 0;
      MPI_Recv(&tick_id, 1, MPI_UNSIGNED, rIng, TAG_FEAT, MPI_COMM_WORLD, &st);
      MPI_Recv(&cnt, 1, MPI_INT, rIng, TAG_FEAT, MPI_COMM_WORLD, &st);
      samples.resize(cnt);
      MPI_Recv(samples.data(), cnt * sizeof(SensorSample), MPI_BYTE, rIng, TAG_FEAT, MPI_COMM_WORLD, &st);

      // Map features
      agg.map_features(samples, feats);

      // Thin features according to BP level by stride sampling
      thin.clear();
      thin.reserve((feats.size() + stride - 1) / stride);
      for (size_t i = 0; i < feats.size(); i += stride)
        thin.push_back(feats[i]);

      // Scatter contiguous slices to predictors
      int per = (int)thin.size() / P;
      int cursor = 0;
      for (int p = 0; p < P; ++p)
      {
        int begin = cursor;
        int end = (p == P - 1) ? (int)thin.size() : (cursor + per);
        int n = end - begin;
        MPI_Send(&tick_id, 1, MPI_UNSIGNED, p + 1, TAG_FEAT, MPI_COMM_WORLD);
        MPI_Send(&n, 1, MPI_INT, p + 1, TAG_FEAT, MPI_COMM_WORLD);
        if (n > 0)
          MPI_Send(thin.data() + begin, n * sizeof(Features), MPI_BYTE, p + 1, TAG_FEAT, MPI_COMM_WORLD);
        cursor = end;
      }
    }
  }

  // ---------------- Predictor workers ----------------
  else if (rank >= 1 && rank <= P)
  {
    Predictor pred(pcfg);
    std::vector<Features> feats;
    std::vector<Prediction> preds;

    for (uint32_t t = 0; t < TOTAL_TICKS; ++t)
    {
      MPI_Status st;
      uint32_t tick_id;
      int n = 0;
      MPI_Recv(&tick_id, 1, MPI_UNSIGNED, P + 1, TAG_FEAT, MPI_COMM_WORLD, &st);
      MPI_Recv(&n, 1, MPI_INT, P + 1, TAG_FEAT, MPI_COMM_WORLD, &st);

      feats.resize(n);
      if (n > 0)
        MPI_Recv(feats.data(), n * sizeof(Features), MPI_BYTE, P + 1, TAG_FEAT, MPI_COMM_WORLD, &st);

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
      int outn = (int)preds.size();
      MPI_Send(&outn, 1, MPI_INT, 0, TAG_PRED, MPI_COMM_WORLD);
      if (outn > 0)
        MPI_Send(preds.data(), outn * sizeof(Prediction), MPI_BYTE, 0, TAG_PRED, MPI_COMM_WORLD);
    }
  }

  // ---------------- Controller (real-time orchestrator) ----------------
  else if (rank == rCtrl)
  {
    Controller ctrl(ccfg);

    // Establish a firm timeline: first tick starts a bit in the future
    uint64_t base0 = now_ms();
    uint64_t first_tick_at = base0 + 300; // 300ms from now; gives everyone time to start

    uint32_t misses = 0;

    for (uint32_t t = 0; t < TOTAL_TICKS; ++t)
    {
      uint64_t tick_start = first_tick_at + t * TICK_MS;
      uint64_t tick_end = tick_start + TICK_MS;

      // Align to tick start (optional; keeps cadence clean)
      sleep_until_ms(tick_start);

      std::vector<Prediction> all;
      all.reserve(4096);

      int received_from = 0;
      // Non-blocking gather loop until deadline or all predictors reported
      while (now_ms() < tick_end && received_from < P)
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
        all.resize(off + n);
        if (n > 0)
          MPI_Recv(all.data() + off, n * sizeof(Prediction), MPI_BYTE, st.MPI_SOURCE, TAG_PRED, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        received_from++;
      }

      bool complete = (received_from == P);
      if (!complete)
        misses++;

      // Controller decision (budgeted)
      Deadline dctrl{.start_ms = now_ms(), .budget_ms = BUDGET_CTRL};
      std::vector<PhaseCmd> cmds;
      ctrl.decide(all, cmds, /*predictions_complete*/ complete);
      (void)dctrl; // reserved for future accounting

      // Quick visibility: top1 and running miss ratio
      if (!all.empty())
      {
        std::nth_element(all.begin(), all.begin() + 1, all.end(),
                         [](const Prediction &a, const Prediction &b)
                         { return a.congestion_60s > b.congestion_60s; });
      }
      double miss_ratio = (double)misses / (double)(t + 1);
      std::printf("[CTRL] tick %2u | slices %d/%d | preds=%zu | top0=%u | miss-ratio=%.2f\n",
                  t, complete ? P : received_from, P, all.size(), all.empty() ? 9999 : all[0].junction, miss_ratio);

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
      sleep_until_ms(tick_end);
    }
  }

  MPI_Finalize();
  return 0;
}