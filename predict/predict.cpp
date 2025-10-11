#include "predict/predict.h"
#include <cstdlib>
#include <cmath>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

// ---- OpenCL minimal host (optional) ----
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

struct Predictor::ClCtx
{
  cl_platform_id platform{};
  cl_device_id device{};
  cl_context ctx{};
  cl_command_queue q{};
  cl_program prog{};
  cl_kernel kern{};
  cl_mem dX{}, dW{}, dO{};
  int B_cap = 0;
};

static const char *KERNEL_SRC = R"CLC(
__kernel void infer_linear(__global const float* X,
                           __global const float* W,
                           const float bias,
                           __global float* out,
                           int F) {
  int i = get_global_id(0);
  float acc = bias;
  for (int j=0;j<F;++j) acc += X[i*F + j]*W[j];
  out[i] = clamp(1.f/(1.f+exp(-acc)), 0.f, 1.f);
}
)CLC";

Predictor::Predictor(const PredConfig &c) : cfg_(c) { init_opencl_if_possible(); }

void Predictor::init_opencl_if_possible()
{
  if (!cfg_.prefer_opencl)
    return;
  cl_uint np = 0;
  if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0)
    return;
  std::vector<cl_platform_id> plats(np);
  clGetPlatformIDs(np, plats.data(), nullptr);
  for (auto p : plats)
  {
    cl_uint nd = 0;
    if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || nd == 0)
      continue;
    std::vector<cl_device_id> devs(nd);
    clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr);
    cl_ = new ClCtx();
    cl_->platform = p;
    cl_->device = devs[0];
    cl_int err = 0;
#if defined(CL_VERSION_2_0)
    cl_->ctx = clCreateContext(nullptr, 1, &cl_->device, nullptr, nullptr, &err);
    cl_->q = clCreateCommandQueueWithProperties(cl_->ctx, cl_->device, 0, &err);
#else
    cl_->ctx = clCreateContext(nullptr, 1, &cl_->device, nullptr, nullptr, &err);
    cl_->q = clCreateCommandQueue(cl_->ctx, cl_->device, 0, &err);
#endif
    const char *src = KERNEL_SRC;
    size_t len = std::strlen(KERNEL_SRC);
    cl_->prog = clCreateProgramWithSource(cl_->ctx, 1, &src, &len, &err);
    if (clBuildProgram(cl_->prog, 1, &cl_->device, "", nullptr, nullptr) != CL_SUCCESS)
    {
      delete cl_;
      cl_ = nullptr;
      continue;
    }
    cl_->kern = clCreateKernel(cl_->prog, "infer_linear", &err);
    has_cl_ = true;
    break;
  }
}

void Predictor::cpu_predict(const std::vector<Features> &feats, std::vector<Prediction> &out)
{
  out.resize(feats.size());
  // small linear model on f0..f5
  const float W[6] = {0.06f, 0.04f, -0.05f, 0.08f, 0.02f, 0.02f};
  const float bias = 0.1f;

  // --- extra CPU work knob (per-feature), controlled by env PRED_WORK ---
  static int WORK_K = []
  {
    const char *s = std::getenv("PRED_WORK");
    int v = s ? std::atoi(s) : 0;
    return v < 0 ? 0 : v;
  }();

  auto extra_work = [&](float seed)
  {
    // deterministic per feature; do math that the compiler won't fold away
    double x = 0.001 + seed * 0.0001;
    for (int k = 0; k < WORK_K; ++k)
    {
      x = std::sin(x) * std::cos(x) + x * 1.000001;
    }
    volatile double sink = x;
    (void)sink; // prevents over-optimization
  };
  // ----------------------------------------------------------------------

#pragma omp parallel for
  for (int i = 0; i < (int)feats.size(); ++i)
  {
    if (WORK_K > 0)
      extra_work(feats[i].f[0]); // optional heavy work (doesn't change output)

    float z = bias;
    for (int j = 0; j < 6; ++j)
      z += feats[i].f[j] * W[j];
    float y = 1.f / (1.f + std::exp(-z));
    out[i] = Prediction{feats[i].ts_ms, feats[i].junction, y};
  }
}

void Predictor::predict_batch(const std::vector<Features> &feats, std::vector<Prediction> &out)
{
  if (!has_cl_ || feats.empty())
  {
    cpu_predict(feats, out);
    return;
  }
  // OpenCL path
  const int F = 6; // use f0..f5
  const float W[6] = {0.06f, 0.04f, -0.05f, 0.08f, 0.02f, 0.02f};
  const float bias = 0.1f;

  cl_int err = 0;
  int B = (int)feats.size();
  std::vector<float> X(B * F), O(B);
  for (int i = 0; i < B; ++i)
    for (int j = 0; j < F; ++j)
      X[i * F + j] = feats[i].f[j];

  if (cl_->B_cap < B)
  {
    if (cl_->dX)
    {
      clReleaseMemObject(cl_->dX);
      clReleaseMemObject(cl_->dW);
      clReleaseMemObject(cl_->dO);
    }
    cl_->dX = clCreateBuffer(cl_->ctx, CL_MEM_READ_ONLY, sizeof(float) * B * F, nullptr, &err);
    cl_->dW = clCreateBuffer(cl_->ctx, CL_MEM_READ_ONLY, sizeof(float) * F, nullptr, &err);
    cl_->dO = clCreateBuffer(cl_->ctx, CL_MEM_WRITE_ONLY, sizeof(float) * B, nullptr, &err);
    cl_->B_cap = B;
  }

  clEnqueueWriteBuffer(cl_->q, cl_->dX, CL_TRUE, 0, sizeof(float) * B * F, X.data(), 0, nullptr, nullptr);
  clEnqueueWriteBuffer(cl_->q, cl_->dW, CL_TRUE, 0, sizeof(float) * F, W, 0, nullptr, nullptr);

  clSetKernelArg(cl_->kern, 0, sizeof(cl_mem), &cl_->dX);
  clSetKernelArg(cl_->kern, 1, sizeof(cl_mem), &cl_->dW);
  clSetKernelArg(cl_->kern, 2, sizeof(float), &bias);
  clSetKernelArg(cl_->kern, 3, sizeof(cl_mem), &cl_->dO);
  clSetKernelArg(cl_->kern, 4, sizeof(int), &F);

  size_t g = B;
  clEnqueueNDRangeKernel(cl_->q, cl_->kern, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
  clFinish(cl_->q);
  clEnqueueReadBuffer(cl_->q, cl_->dO, CL_TRUE, 0, sizeof(float) * B, O.data(), 0, nullptr, nullptr);

  out.resize(B);
  for (int i = 0; i < B; ++i)
    out[i] = Prediction{feats[i].ts_ms, feats[i].junction, O[i]};
}