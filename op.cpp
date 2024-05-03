#include "op.h"

#include <torch/torch.h>
#include <vector>
#include <iostream>

void ffkan( float* x, float* coeff, float* bias, int bs, int inputdim, int outputdim, int gridsize, float* out );

void ffkan_b(float *x, float *xb,
             float *coeff, float *coeffb,
              float *bias, float *biasb,
               int bs, int inputdim, int outputdim, int gridsize,
                float*out, float *outb);
                
torch::Tensor ffKANForward(torch::Tensor x, torch::Tensor coeff,torch::Tensor bias) 
{
int bs = x.size(0);
int inputdim = x.size(1);
//coeff shape (2,inputdim,outputdim,gridsize)
if(coeff.size(0) != 2)
{
  std::cout<< "coeff.size(0) != 2 " << coeff.size(0) << " != 2" << std::endl;
  throw;
}
if(coeff.size(1) != inputdim)
{
  std::cout<< "coeff.size(1) != inputdim " << coeff.size(1) << " != " << inputdim << std::endl;
  throw;
}

int outputdim = coeff.size(2);
int gridsize = coeff.size(3);



torch::Tensor out = torch::zeros({bs,outputdim});

ffkan(x.data_ptr<float>(), coeff.data_ptr<float>(),bias.data_ptr<float>(),bs,inputdim,outputdim,gridsize,out.data_ptr<float>() );

return out;
}

std::vector<torch::Tensor> ffKANBackward(torch::Tensor x, torch::Tensor coeff,torch::Tensor bias, torch::Tensor outb)
{

int bs = x.size(0);
int inputdim = x.size(1);
//coeff shape (2,inputdim,outputdim,gridsize)
if(coeff.size(0) != 2)
{
  std::cout<< "coeff.size(0) != 2 " << coeff.size(0) << " != 2" << std::endl;
  throw;
}
if(coeff.size(1) != inputdim)
{
  std::cout<< "coeff.size(1) != inputdim " << coeff.size(1) << " != " << inputdim << std::endl;
  throw;
}

int outputdim = coeff.size(2);
int gridsize = coeff.size(3);

auto options =torch::TensorOptions().dtype(torch::kFloat32);
torch::Tensor xb = torch::zeros({bs,inputdim}, options);
torch::Tensor coeffb = torch::zeros({2,inputdim,outputdim,gridsize}, options);
torch::Tensor biasb = torch::zeros({outputdim}, options);


ffkan_b(x.data_ptr<float>(),xb.data_ptr<float>(),
        coeff.data_ptr<float>(),coeffb.data_ptr<float>(),
        bias.data_ptr<float>(),biasb.data_ptr<float>(),
        bs,inputdim,outputdim,gridsize,
        NULL,outb.data_ptr<float>() );

return {xb,coeffb,biasb};



}