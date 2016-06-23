/*!
 *  Copyright (c) 2015 by Contributors
 * \file adadelta-inl.h
 * \brief AdaDelta optimizer interface of mxnet.
 * \author Xin Li
 */
#ifndef MXNET_OPTIMIZER_ADADELTA_INL_H_
#define MXNET_OPTIMIZER_ADADELTA_INL_H_

#include <mshadow/tensor.h>
#include <mxnet/optimizer.h>
#include <dmlc/parameter.h>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include "../operator/mshadow_op.h"

namespace mxnet {
namespace opt {

  struct AdaGradParam : public dmlc::Parameter<AdaGradParam> {
  float rho;
  float eps;
  float wd;
  float rescale_grad;
  float clip_gradient;
  DMLC_DECLARE_PARAMETER(AdaGradParam) {
    DMLC_DECLARE_FIELD(rho)
      .set_default(0.9f)
      .describe("Decay rate for both squared gradients and delta x");
    DMLC_DECLARE_FIELD(eps)
      .set_default(1e-5)
      .describe("The constant as described in the thesis");
    DMLC_DECLARE_FIELD(rescale_grad)
    .set_default(1.0f)
    .describe("rescale gradient as grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
    .set_default(-1.0f)
    .describe("If greater than 0, clip gradient to "
              "grad = max(min(grad, -clip_gradient), clip_gradient). "
              "Otherwise turned off.");
  }
};

template<typename xpu>
void adagrad_update(RunContext ctx, TBlob weight, const TBlob grad,
                     TBlob history, float lr, float wd, const AdaGradParam& param) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet::op;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  Tensor<xpu, 2> weight2d = weight.FlatTo2D<xpu, real_t>(s);
  Tensor<xpu, 2> grad2d = grad.FlatTo2D<xpu, real_t>(s);
  Tensor<xpu, 2> h2d = history.FlatTo2D<xpu, real_t>(s);
  h2d += grad2d * grad2d;
  auto rms = F<mshadow_op::square_root>(h2d + param.eps);
  auto ratio = F<mshadow::op::div>(grad2d, rms);
  auto decayed = wd * weight2d;
  auto merge = ratio + decayed;
  weight2d -= lr * merge;
}

void call_adagrad_update_cpu(RunContext ctx, TBlob weight, const TBlob grad,
                TBlob history, float lr, float wd, const AdaGradParam& param);
#if MXNET_USE_CUDA
void call_adagrad_update_gpu(RunContext ctx, TBlob weight, const TBlob grad,
                TBlob history, float lr, float wd, const AdaGradParam& param);
#endif  // MXNET_USE_CUDA

#if DMLC_USE_CXX11

class AdaGradOpt : public Optimizer {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  void CreateState(const int index, const NDArray *weight) override {
    if (history_.find(index) == history_.end()) {
      history_[index] = NDArray(weight->shape(), weight->ctx());
      history_[index] = 0.0f;
    }
  }

  void Update(const int index, NDArray *weight,
              const NDArray *grad, const float lr, const float wd) override {
    NDArray w = *weight, g = *grad;
    CreateState(index, weight);
    switch (w.ctx().dev_type) {
     case Context::kCPU:
     case Context::kCPUPinned:
       Engine::Get()->PushSync([this, index, w, g, lr, wd](RunContext ctx) {
         call_adagrad_update_cpu(ctx, w.data(), g.data(),
             history_[index].data(), lr, wd, param_);
       }, w.ctx(), { g.var() }, { w.var(), history_[index].var() },
       FnProperty::kNormal);
       break;
     case Context::kGPU:
#if MXNET_USE_CUDA
       Engine::Get()->PushSync([this, index, w, g, lr, wd](RunContext ctx) {
         call_adagrad_update_gpu(ctx, w.data(), g.data(),
             history_[index].data(), lr, wd, param_);
       }, w.ctx(), { g.var() }, { w.var(), history_[index].var() },
       FnProperty::kNormal);
       break;
#else
       LOG(FATAL) << "Please compile with CUDA enabled for cuda features";
#endif  // MXNET_USE_CUDA
     default:
      LOG(FATAL) << "Unsupported device type for adagrad optimizer: " << w.ctx().dev_type;
    }
  }

 private:
  AdaGradParam param_;
  std::map<int, NDArray> history_;
};

#endif  // DMLC_USE_CXX11

}  // namespace opt
}  // namespace mxnet
#endif  // MXNET_OPTIMIZER_ADADELTA_INL_H_
