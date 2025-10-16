// predict/predict.cpp
#include "predict/predict.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

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
#if defined(CL_VERSION_2_0)
  cl_command_queue q{};
#else
  cl_command_queue q{};
#endif
  cl_program prog{};
  cl_kernel kern{};
  cl_mem dX{}, dW{}, dO{};
  int B_cap = 0; // current buffer capacity in batch items
  int F_cap = 0; // feature width capacity (should be 6 here)
};

// Same kernel as predict/kernels.cl so the code is self-contained.
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

// --- small helper: print OpenCL build log on failure ---
static void cl_print_build_log(cl_program prog, cl_device_id dev)
{
  size_t sz = 0;
  if (clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &sz) == CL_SUCCESS && sz > 1)
  {
    std::string log(sz, '\0');
    if (clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, sz, log.data(), nullptr) == CL_SUCCESS)
    {
      std::fprintf(stderr, "[OpenCL] build log:\n%s\n", log.c_str());
    }
  }
}

Predictor::Predictor(const PredConfig &c) : cfg_(c) { init_opencl_if_possible(); }

Predictor::~Predictor()
{
  if (!cl_)
    return;
  if (cl_->kern)
    clReleaseKernel(cl_->kern);
  if (cl_->prog)
    clReleaseProgram(cl_->prog);
  if (cl_->dX)
    clReleaseMemObject(cl_->dX);
  if (cl_->dW)
    clReleaseMemObject(cl_->dW);
  if (cl_->dO)
    clReleaseMemObject(cl_->dO);
  if (cl_->q)
    clReleaseCommandQueue(cl_->q);
  if (cl_->ctx)
    clReleaseContext(cl_->ctx);
  delete cl_;
  cl_ = nullptr;
  has_cl_ = false;
}

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
    if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr) != CL_SUCCESS)
      continue;

    cl_int err = CL_SUCCESS;
    auto *ctx = new ClCtx();
    ctx->platform = p;
    ctx->device = devs[0];

    ctx->ctx = clCreateContext(nullptr, 1, &ctx->device, nullptr, nullptr, &err);
    if (!ctx->ctx || err != CL_SUCCESS)
    {
      delete ctx;
      continue;
    }

#if defined(CL_VERSION_2_0)
    ctx->q = clCreateCommandQueueWithProperties(ctx->ctx, ctx->device, nullptr, &err);
#else
    ctx->q = clCreateCommandQueue(ctx->ctx, ctx->device, 0, &err);
#endif
    if (!ctx->q || err != CL_SUCCESS)
    {
      clReleaseContext(ctx->ctx);
      delete ctx;
      continue;
    }

    // Create & build program
    const char *src = KERNEL_SRC;
    size_t len = std::strlen(KERNEL_SRC);
    ctx->prog = clCreateProgramWithSource(ctx->ctx, 1, &src, &len, &err);
    if (!ctx->prog || err != CL_SUCCESS)
    {
      clReleaseCommandQueue(ctx->q);
      clReleaseContext(ctx->ctx);
      delete ctx;
      continue;
    }
    if (clBuildProgram(ctx->prog, 1, &ctx->device, "", nullptr, nullptr) != CL_SUCCESS)
    {
      cl_print_build_log(ctx->prog, ctx->device);
      clReleaseProgram(ctx->prog);
      clReleaseCommandQueue(ctx->q);
      clReleaseContext(ctx->ctx);
      delete ctx;
      continue;
    }

    ctx->kern = clCreateKernel(ctx->prog, "infer_linear", &err);
    if (!ctx->kern || err != CL_SUCCESS)
    {
      clReleaseProgram(ctx->prog);
      clReleaseCommandQueue(ctx->q);
      clReleaseContext(ctx->ctx);
      delete ctx;
      continue;
    }

    cl_ = ctx;
    has_cl_ = true;
    break; // success
  }
}

void Predictor::cpu_predict(const std::vector<Features> &feats, std::vector<Prediction> &out)
{
  out.resize(feats.size());

  // tiny linear model over f0..f5
  constexpr int F = 6;
  const float W[F] = {0.06f, 0.04f, -0.05f, 0.08f, 0.02f, 0.02f};
  const float bias = 0.1f;

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(feats.size()); ++i)
  {
    float z = bias;
    for (int j = 0; j < F; ++j)
      z += feats[i].f[j] * W[j];
    const float y = 1.f / (1.f + std::exp(-z));
    out[i] = Prediction{feats[i].ts_ms, feats[i].junction, std::min(std::max(y, 0.f), 1.f)};
  }
}

void Predictor::predict_batch(const std::vector<Features> &feats, std::vector<Prediction> &out)
{
  if (!has_cl_ || feats.empty())
  {
    cpu_predict(feats, out);
    return;
  }

  // OpenCL path (same model as CPU path)
  constexpr int F = 6;
  const float W[F] = {0.06f, 0.04f, -0.05f, 0.08f, 0.02f, 0.02f};
  const float bias = 0.1f;

  const int B = static_cast<int>(feats.size());
  std::vector<float> X;
  X.resize(static_cast<size_t>(B) * F);

  for (int i = 0; i < B; ++i)
    for (int j = 0; j < F; ++j)
      X[i * F + j] = feats[i].f[j];

  cl_int err = CL_SUCCESS;

  // (Re)allocate buffers if capacity is insufficient
  if (cl_->B_cap < B || cl_->F_cap < F)
  {
    if (cl_->dX)
      clReleaseMemObject(cl_->dX);
    if (cl_->dW)
      clReleaseMemObject(cl_->dW);
    if (cl_->dO)
      clReleaseMemObject(cl_->dO);

    cl_->dX = clCreateBuffer(cl_->ctx, CL_MEM_READ_ONLY, sizeof(float) * B * F, nullptr, &err);
    if (!cl_->dX || err != CL_SUCCESS)
    {
      cpu_predict(feats, out);
      return;
    }

    cl_->dW = clCreateBuffer(cl_->ctx, CL_MEM_READ_ONLY, sizeof(float) * F, nullptr, &err);
    if (!cl_->dW || err != CL_SUCCESS)
    {
      cpu_predict(feats, out);
      return;
    }

    cl_->dO = clCreateBuffer(cl_->ctx, CL_MEM_WRITE_ONLY, sizeof(float) * B, nullptr, &err);
    if (!cl_->dO || err != CL_SUCCESS)
    {
      cpu_predict(feats, out);
      return;
    }

    cl_->B_cap = B;
    cl_->F_cap = F;
  }

  // Upload X, W
  if (clEnqueueWriteBuffer(cl_->q, cl_->dX, CL_TRUE, 0, sizeof(float) * B * F, X.data(), 0, nullptr, nullptr) != CL_SUCCESS)
  {
    cpu_predict(feats, out);
    return;
  }
  if (clEnqueueWriteBuffer(cl_->q, cl_->dW, CL_TRUE, 0, sizeof(float) * F, W, 0, nullptr, nullptr) != CL_SUCCESS)
  {
    cpu_predict(feats, out);
    return;
  }

  // Set args
  clSetKernelArg(cl_->kern, 0, sizeof(cl_mem), &cl_->dX);
  clSetKernelArg(cl_->kern, 1, sizeof(cl_mem), &cl_->dW);
  clSetKernelArg(cl_->kern, 2, sizeof(float), &bias);
  clSetKernelArg(cl_->kern, 3, sizeof(cl_mem), &cl_->dO);
  clSetKernelArg(cl_->kern, 4, sizeof(int), const_cast<int *>(&F));

  // Launch
  size_t g = static_cast<size_t>(B);
  if (clEnqueueNDRangeKernel(cl_->q, cl_->kern, 1, nullptr, &g, nullptr, 0, nullptr, nullptr) != CL_SUCCESS)
  {
    cpu_predict(feats, out);
    return;
  }
  clFinish(cl_->q);

  // Download output
  std::vector<float> O(B);
  if (clEnqueueReadBuffer(cl_->q, cl_->dO, CL_TRUE, 0, sizeof(float) * B, O.data(), 0, nullptr, nullptr) != CL_SUCCESS)
  {
    cpu_predict(feats, out);
    return;
  }

  // Stitch predictions
  out.resize(B);
  for (int i = 0; i < B; ++i)
  {
    float y = O[i];
    if (y < 0.f)
      y = 0.f;
    else if (y > 1.f)
      y = 1.f;
    out[i] = Prediction{feats[i].ts_ms, feats[i].junction, y};
  }
}