#pragma once

#include "AuxiliaryData.h"
#include "WorkingMemory.h"

#include <torch/nn.h>

namespace dorado::nn {

struct LinearLayerImpl : torch::nn::Module {
    virtual at::Tensor forward(at::Tensor) = 0;

#if DORADO_CUDA_BUILD
    virtual void reserve_working_memory(WorkingMemory &, const AuxiliaryData *) = 0;
    virtual void run_koi(WorkingMemory &, const AuxiliaryData *) = 0;
#endif
};

TORCH_MODULE(LinearLayer);

}  // namespace dorado::nn
