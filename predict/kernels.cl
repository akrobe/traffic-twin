// predict/kernels.cl
// (Optional) Mirrors the embedded kernel in predict.cpp so the repo
// clearly documents the GPU path. Not required at runtime because
// predict.cpp embeds the same source as a string.

__kernel void infer_linear(__global const float* X,
                           __global const float* W,
                           const float bias,
                           __global float* out,
                           int F) {
  int i = get_global_id(0);
  float acc = bias;
  for (int j = 0; j < F; ++j) {
    acc += X[i * F + j] * W[j];
  }
  // Logistic and clamp to [0,1]
  out[i] = clamp(1.0f / (1.0f + exp(-acc)), 0.0f, 1.0f);
}