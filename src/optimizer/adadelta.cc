/*!
 * Copyright (c) 2015 by Contributors
 * \file sgd.cc
 * \brief sgd optimizer
*/
#include <mxnet/ndarray.h>
#include "./adadelta-inl.h"


namespace mxnet {
namespace opt {

void call_adadelta_update_cpu(RunContext ctx, TBlob weight, const TBlob grad,
    TBlob acc_g, TBlob acc_delta, float wd, const AdaDeltaParam& param) {
  adadelta_update<cpu>(ctx, weight, grad, acc_g, acc_delta, wd, param);
}

DMLC_REGISTER_PARAMETER(AdaDeltaParam);

MXNET_REGISTER_OPTIMIZER(adadelta, AdaDeltaOpt)
.describe("adadelta optimizer implemented in C++.");

}  // namespace opt
}  // namespace mxnet
