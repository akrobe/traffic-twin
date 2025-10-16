// predict/predict.h
#pragma once
#include <vector>
#include "common/schema.h"

struct PredConfig
{
  bool prefer_opencl = true;
};

class Predictor
{
public:
  explicit Predictor(const PredConfig &c);
  ~Predictor();
  bool has_opencl() const { return has_cl_; }

  // Predict congestion in 60s horizon [0..1]
  void predict_batch(const std::vector<Features> &feats, std::vector<Prediction> &out);

private:
  PredConfig cfg_;
  bool has_cl_ = false;

  // Minimal OpenCL context (opaque in header)
  struct ClCtx;
  ClCtx *cl_ = nullptr;

  void cpu_predict(const std::vector<Features> &feats, std::vector<Prediction> &out);
  void init_opencl_if_possible();
};