#include "aggregate/aggregate.h"
#include <algorithm>
#include <cmath>
#ifdef _OPENMP
  #include <omp.h>
#endif

Aggregator::Aggregator(const AggConfig& c) : cfg_(c), ema_q_(c.junctions, 0.f) {}

void Aggregator::map_features(const std::vector<SensorSample>& samples, std::vector<Features>& out) {
  out.clear();
  out.resize(cfg_.junctions);
  const float alpha = 0.15f;

  // aggregate per junction
  #pragma omp parallel for
  for (int j = 0; j < (int)cfg_.junctions; ++j) {
    float sum_q = 0.f, sum_a = 0.f, sum_v = 0.f;
    int cnt = 0;
    for (uint32_t l=0; l<cfg_.lanes_per; ++l) {
      const auto& s = samples[j*cfg_.lanes_per + l];
      sum_q += s.q_len; sum_a += s.arrivals; sum_v += s.avg_speed; cnt++;
    }
    float mq = sum_q / cnt;
    float ma = sum_a / cnt / 10.f;
    float mv = sum_v / cnt / 10.f;

    // simple EWMA on queue
    ema_q_[j] = alpha*mq + (1.f-alpha)*ema_q_[j];

    Features f{};
    f.ts_ms = samples[j*cfg_.lanes_per].ts_ms;
    f.junction = (uint16_t)j;
    f.f[0]=mq; f.f[1]=ma; f.f[2]=mv; f.f[3]=ema_q_[j];
    f.f[4]=std::sin((f.ts_ms/1000)%86400 * (2*M_PI/86400.0)); // time of day sin
    f.f[5]=std::cos((f.ts_ms/1000)%86400 * (2*M_PI/86400.0)); // time of day cos
    for (int k=6;k<MAX_FEATURES;++k) f.f[k]=0.f;
    out[j]=f;
  }
}

void Aggregator::reduce_topN(const std::vector<Features>& feats, int N, std::vector<uint16_t>& out_top) {
  std::vector<std::pair<float,uint16_t>> score;
  score.reserve(feats.size());
  for (const auto& f: feats) score.emplace_back(f.f[0] + 0.5f*f.f[3], f.junction);
  if (N > (int)score.size()) N = (int)score.size();
  std::nth_element(score.begin(), score.begin()+N, score.end(),
                   [](auto&a, auto&b){return a.first>b.first;});
  out_top.clear();
  out_top.reserve(N);
  for (int i=0;i<N;++i) out_top.push_back(score[i].second);
  std::sort(out_top.begin(), out_top.end());
}