#pragma once
#include <vector>
#include "common/schema.h"

struct PredConfig {
  bool prefer_opencl = true;
};

class Predictor {
public:
  explicit Predictor(const PredConfig& c);
  bool has_opencl() const { return has_cl_; }

  // CPU fallback or OpenCL: predict congestion in 60s horizon [0..1]
  void predict_batch(const std::vector<Features>& feats, std::vector<Prediction>& out);

private:
  PredConfig cfg_;
  bool has_cl_ = false;

  // OpenCL handles (opaque forward decl to avoid heavy headers)
  struct ClCtx;
  ClCtx* cl_ = nullptr;

  void cpu_predict(const std::vector<Features>& feats, std::vector<Prediction>& out);
  void init_opencl_if_possible();
};