/*!
 * Copyright (c) 2015 by Contributors
 * \file sgd.cc
 * \brief sgd optimizer
*/
#include <mxnet/ndarray.h>
#include "./adagrad-inl.h"


namespace mxnet {
namespace opt {

void call_adagrad_update_cpu(RunContext ctx, TBlob weight, const TBlob grad,
    TBlob history, float lr, float wd, const AdaGradParam& param) {
  adagrad_update<cpu>(ctx, weight, grad, history, lr, wd, param);
}

DMLC_REGISTER_PARAMETER(AdaGradParam);

MXNET_REGISTER_OPTIMIZER(adagrad, AdaGradOpt)
.describe("adadelta optimizer implemented in C++.");

}  // namespace opt
}  // namespace mxnet
