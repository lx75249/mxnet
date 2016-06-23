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

  struct AdaDeltaParam : public dmlc::Parameter<AdaDeltaParam> {
  float rho;
  float eps;
  float wd;
  float rescale_grad;
  float clip_gradient;
  DMLC_DECLARE_PARAMETER(AdaDeltaParam) {
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

struct adadelta_clip {
  MSHADOW_XINLINE static real_t Map(real_t x, real_t bound) {
    if (x > bound) {
      return bound;
    } else if (x < -bound) {
      return -bound;
    } else {
      return x;
    }
  }
};

template<typename xpu>
void adadelta_update(RunContext ctx, TBlob weight, const TBlob grad,
                     TBlob acc_g, TBlob acc_delta,
                     float wd, const AdaDeltaParam& param) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mxnet::op;
  Stream<xpu>* s = ctx.get_stream<xpu>();
  Tensor<xpu, 2> weight2d = weight.FlatTo2D<xpu, real_t>(s);
  Tensor<xpu, 2> grad2d = grad.FlatTo2D<xpu, real_t>(s);
  Tensor<xpu, 2> g2 = acc_g.FlatTo2D<xpu, real_t>(s);
  Tensor<xpu, 2> delta = acc_delta.FlatTo2D<xpu, real_t>(s);
  std::unique_ptr<real_t> tmp(new real_t[delta.shape_[0] * delta.shape_[1]]);
  TBlob Zero(tmp.get(), acc_delta.shape_, acc_delta.dev_mask_);
  Tensor<xpu, 2> zero = Zero.FlatTo2D<xpu, real_t>(s);
  if (param.clip_gradient >= 0.0f) {
    weight2d -= param.rescale_grad*F<adadelta_clip>(grad2d, param.clip_gradient) +
                wd*weight2d;
  } else {
    auto t2 = delta + ScalarExp<real_t>(param.eps);
    LG << "eps = " << param.eps;
    zero = t2;
    LG << "delta + eps = " << zero.dptr_[0];
    /*
    g2 *= param.rho;
    g2 += (1.0f - param.rho) * F<mshadow_op::square>(grad2d);
    zero = 0 + grad2d;
    LG << "grad2d = " << zero.dptr_[0];
    zero = 0 + delta;
    LG << "delta =" << zero.dptr_[0];
    auto t2 = delta + param.eps;
    LG << "eps = " << param.eps;
    zero = 0 + t2;
    LG << "delta + eps = " << zero.dptr_[0];
    zero = 0 + g2;
    LG << "g2 = " << zero.dptr_[0];
    auto g2e = g2 + param.eps;
    zero = 0 + g2e;
    LG << "g2 + eps = " << zero.dptr_[0];
    auto rms_delta = F<mshadow_op::square_root>(t2);
    zero = 0 + rms_delta;
    LG << zero.dptr_[0];
    auto rms_g = F<mshadow_op::square_root>(g2e);
    zero = 0 + rms_g;
    LG << zero.dptr_[0];
    auto ratio = F<mshadow::op::div>(rms_delta, rms_g);
    zero = 0 + ratio;
    LG << zero.dptr_[0];
    auto current_delta = F<mshadow::op::mul>(ratio, grad2d);
    zero *= 0.0f;
    zero += current_delta;
    LG << zero.dptr_[0];
    auto decayed = wd*weight2d;
    //auto diff = current_delta + decayed;
    //weight2d -= diff;
    weight2d -= decayed;
    weight2d -= current_delta;
    auto cdelta2 = F<mshadow_op::square>(current_delta);
    delta *= param.rho;
    delta += (1 - param.rho) * cdelta2;
    */
  }
}

void call_adadelta_update_cpu(RunContext ctx, TBlob weight, const TBlob grad,
                TBlob acc_g, TBlob acc_delta, float wd, const AdaDeltaParam& param);
#if MXNET_USE_CUDA
void call_adadelta_update_gpu(RunContext ctx, TBlob weight, const TBlob grad,
                TBlob acc_g, TBlob acc_delta, float wd, const AdaDeltaParam& param);
#endif  // MXNET_USE_CUDA

#if DMLC_USE_CXX11

class AdaDeltaOpt : public Optimizer {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  void CreateState(const int index, const NDArray *weight) override {
    if (acc_g_.find(index) == acc_g_.end()) {
      acc_g_[index] = NDArray(weight->shape(), weight->ctx());
      acc_g_[index] = 0.0f;
      acc_delta_[index] = NDArray(weight->shape(), weight->ctx());
      acc_delta_[index] = 0.0f;
    }
  }

  void Update(const int index, NDArray *weight,
              const NDArray *grad, const float lr, const float wd) override {
    NDArray w = *weight, g = *grad;
    CreateState(index, weight);
    switch (w.ctx().dev_type) {
     case Context::kCPU:
     case Context::kCPUPinned:
       Engine::Get()->PushSync([this, index, w, g, wd](RunContext ctx) {
         call_adadelta_update_cpu(ctx, w.data(), g.data(),
             acc_g_[index].data(), acc_delta_[index].data(), wd, param_);
       }, w.ctx(), { g.var() }, { w.var(), acc_g_[index].var(), acc_delta_[index].var() },
       FnProperty::kNormal);
       break;
     case Context::kGPU:
#if MXNET_USE_CUDA
       Engine::Get()->PushSync([this, index, w, g, wd](RunContext ctx) {
         call_adadelta_update_gpu(ctx, w.data(), g.data(),
           acc_g_[index].data(), acc_delta_[index].data(), wd, param_);
       }, w.ctx(), { g.var() }, { w.var(), acc_g_[index].var(), acc_delta_[index].var() },
       FnProperty::kNormal);
       break;
#else
       LOG(FATAL) << "Please compile with CUDA enabled for cuda features";
#endif  // MXNET_USE_CUDA
     default:
      LOG(FATAL) << "Unsupported device type for adadelta optimizer: " << w.ctx().dev_type;
    }
  }

 private:
  AdaDeltaParam param_;
  std::map<int, NDArray> acc_g_, acc_delta_;
};

#endif  // DMLC_USE_CXX11

}  // namespace opt
}  // namespace mxnet
#endif  // MXNET_OPTIMIZER_ADADELTA_INL_H_
