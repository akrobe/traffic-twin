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

// ------------ Real-time budgets (ms) ------------
static constexpr uint32_t TICK_MS = 250;     // tighter 250 ms control loop
static constexpr uint32_t BUDGET_PRED = 120; // per-slice prediction budget
static constexpr uint32_t BUDGET_CTRL = 80;  // controller decision time

// ------------ Back-pressure tag and helpers ------------
#ifndef TAG_BP
#define TAG_BP 9001
#endif

static inline int stride_for_level(int level)
{
  if (level < 0)
    level = 0;
  if (level > 3)
    level = 3;
  return 1 << level; // 1,2,4,8
}

static void send_bp_to_agg(int rAgg, int level)
{
  MPI_Send(&level, 1, MPI_INT, rAgg, TAG_BP, MPI_COMM_WORLD);
}

static void die(const char *m)
{
  std::fprintf(stderr, "FATAL: %s\n", m);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

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
  IngestConfig icfg{.junctions = 20000, .lanes_per = 3, .tick_ms = TICK_MS};
  AggConfig acfg{.junctions = icfg.junctions, .lanes_per = icfg.lanes_per};
  PredConfig pcfg{.prefer_opencl = false}; // force CPU path so PRED_WORK applies
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
    std::vector<Features> feats, thin;

    int bp_level = 0;

    for (uint32_t t = 0; t < 40; ++t)
    {
      // Receive BP updates before shaping work
      bp_level = 0;
      drain_bp_for_rank(rAgg, bp_level);
      int stride = stride_for_level(bp_level);

      // Receive this tick
      MPI_Status st;
      uint32_t tick_id;
      int cnt = 0;
      MPI_Recv(&tick_id, 1, MPI_UNSIGNED, rIng, TAG_FEAT, MPI_COMM_WORLD, &st);
      MPI_Recv(&cnt, 1, MPI_INT, rIng, TAG_FEAT, MPI_COMM_WORLD, &st);
      samples.resize(cnt);
      MPI_Recv(samples.data(), cnt * sizeof(SensorSample), MPI_BYTE, rIng, TAG_FEAT, MPI_COMM_WORLD, &st);

      // Map features and thin by stride
      agg.map_features(samples, feats);
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

    // Banner so we can see env propagation
    const char *w = std::getenv("PRED_WORK");
    std::printf("[PRED rank=%d] prefer_opencl=%d PRED_WORK=%s\n",
                rank, (int)pcfg.prefer_opencl, w ? w : "null");

    for (uint32_t t = 0; t < 40; ++t)
    {
      MPI_Status st;
      uint32_t tick_id;
      int n = 0;
      MPI_Recv(&tick_id, 1, MPI_UNSIGNED, P + 1, TAG_FEAT, MPI_COMM_WORLD, &st);
      MPI_Recv(&n, 1, MPI_INT, P + 1, TAG_FEAT, MPI_COMM_WORLD, &st);
      feats.resize(n);
      if (n > 0)
        MPI_Recv(feats.data(), n * sizeof(Features), MPI_BYTE, P + 1, TAG_FEAT, MPI_COMM_WORLD, &st);

      // Time the slice
      uint64_t t0 = now_ms();
      pred.predict_batch(feats, preds);
      uint32_t dur = (uint32_t)(now_ms() - t0);

      // If we exceeded our slice budget, ask Aggregator to thin next tick
      if (dur > BUDGET_PRED)
      {
        int level = 1; // light request; controller may escalate
        MPI_Send(&level, 1, MPI_INT, P + 1, TAG_BP, MPI_COMM_WORLD);
      }

      // Send to controller
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

    uint64_t base0 = now_ms();
    uint64_t first_tick_at = base0 + 300; // give everyone time to boot
    uint32_t misses = 0;

    for (uint32_t t = 0; t < 40; ++t)
    {
      uint64_t tick_start = first_tick_at + t * TICK_MS;
      uint64_t tick_end = tick_start + TICK_MS;

      // Align to tick start to keep a stable cadence
      sleep_until_ms(tick_start);

      std::vector<Prediction> all;
      all.reserve(4096);
      int received_from = 0;

      // Non-blocking gather until deadline or all predictors reported
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

      // Do a quick decision phase within a budget
      uint64_t c0 = now_ms();
      std::vector<PhaseCmd> cmds;
      ctrl.decide(all, cmds, complete);
      (void)c0; // reserved for future timing if needed

      if (!all.empty())
      {
        std::nth_element(all.begin(), all.begin() + 1, all.end(),
                         [](const Prediction &a, const Prediction &b)
                         { return a.congestion_60s > b.congestion_60s; });
      }
      double miss_ratio = (double)misses / (double)(t + 1);
      std::printf("[CTRL] tick %2u | slices %d/%d | preds=%zu | top0=%u | miss-ratio=%.2f\n",
                  t, complete ? P : received_from, P, all.size(), all.empty() ? 9999 : all[0].junction, miss_ratio);

      // Escalate or relax back-pressure for the next tick
      if (!complete)
      {
        int level = (miss_ratio > 0.20) ? 3 : (miss_ratio > 0.10 ? 2 : 1);
        send_bp_to_agg(P + 1, level);
      }
      else
      {
        send_bp_to_agg(P + 1, 0); // healthy
      }

      // Hold firm cadence
      sleep_until_ms(tick_end);
    }
  }

  MPI_Finalize();
  return 0;
}