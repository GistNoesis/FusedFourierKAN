#ifndef OPGPU_H
#define OPGPU_H

#include <torch/torch.h>
torch::Tensor ffKANGPUForward(torch::Tensor x, torch::Tensor coeff,torch::Tensor bias) ;
std::vector<torch::Tensor> ffKANGPUBackward(torch::Tensor x, torch::Tensor coeff,torch::Tensor bias, torch::Tensor outb);

#endif