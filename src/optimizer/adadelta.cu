/*!
 * Copyright (c) 2015 by Contributors
 * \file adadelta.cu
 * \brief adadelta optimizer
*/
#include "./adadelta-inl.h"

namespace mxnet {
namespace opt {

void call_adadelta_update_gpu(RunContext ctx, TBlob weight, const TBlob grad,
    TBlob acc_g, TBlob acc_delta, float wd, const AdaDeltaParam& param) {
  adadelta_update<gpu>(ctx, weight, grad, acc_g, acc_delta, wd, param);
}

}  // namespace opt
}  // namespace mxnet
