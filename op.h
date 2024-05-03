#ifndef OP_H
#define OP_H

#include <torch/torch.h>
torch::Tensor ffKANForward(torch::Tensor x, torch::Tensor coeff,torch::Tensor bias);
std::vector<torch::Tensor> ffKANBackward(torch::Tensor x, torch::Tensor coeff,torch::Tensor bias, torch::Tensor outb);

#endif