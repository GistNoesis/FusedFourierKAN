import torch as th
from torch.autograd import Function
import os.path

import os.path
import numpy as np

my_path = os.path.abspath(__file__)
parent = os.path.dirname(my_path)
pluginpath = os.path.join(parent, "../build/libfusedFourierKAN.so")
th.ops.load_library(pluginpath)


ffKANForward = th.ops.KAN_ops.ffKANForward
ffKANBackward = th.ops.KAN_ops.ffKANBackward

class FFKANFunction(Function):
    @staticmethod
    def forward(x,coeff,bias):
        
        return ffKANForward(x,coeff,bias)
        
    @staticmethod
    def setup_context(ctx, inputs, output):
        # ctx is a context object that can be used to stash information
        # for backward computation
        #tensor, constant = inputs
        #ctx.constant = constant
        x,coeff,bias = inputs
        
        ctx.save_for_backward(x, coeff, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x,coeff,bias = ctx.saved_tensors
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        xb,coeffb,biasb= ffKANBackward( x,coeff,bias,grad_output)

        return xb,coeffb,biasb

ffKANFunction = FFKANFunction.apply

def target( x, coeff,bias):
    gridsize = coeff.shape[-1]
    k = th.reshape( th.arange(1,gridsize+1,device=x.device),(1,1,1,gridsize))
    xrshp = th.reshape(x,(x.shape[0],1,x.shape[1],1) ) 
    #This should be fused to avoid materializing memory
    c = th.cos( k*xrshp )
    s = th.sin( k*xrshp )
    #We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them 
    y =  th.sum( c*coeff[0:1],(-2,-1)) 
    y += th.sum( s*coeff[1:2],(-2,-1))
    y += th.unsqueeze( bias,0)

    return y

def demo():
    th.manual_seed(42)
    bs = 3
    inputdim = 20
    outputdim = 30
    gridsize = 40
    
    x = th.tensor( th.randn((bs,inputdim)),requires_grad=True)
    coeff =th.tensor( th.randn((2,inputdim,outputdim,gridsize)) / (np.sqrt(inputdim) * np.sqrt(gridsize)) ,requires_grad=True)
    bias = th.tensor( th.randn((outputdim,)),requires_grad=True)

    out = ffKANFunction(x,coeff,bias)

    xtarget = th.tensor(x, requires_grad=True)
    coefftarget = th.tensor(coeff,requires_grad=True)
    biastarget = th.tensor(bias,requires_grad=True)

    permcoeff =  th.permute(coefftarget,(0,2,1,3)) 
    targetout = target(xtarget,permcoeff,biastarget)
    targetloss = th.sum(targetout*targetout)
    targetloss.backward()
    

    print(out.shape)
    loss = th.sum( out*out)

    print( loss )

    loss.backward()

    print( x.grad )
    print( coeff.grad )
    print( bias.grad )

    diffout = th.sum( (out-targetout)**2 )
    print("diffout")
    print(diffout)

    diffxgrad = th.sum( (x.grad-xtarget.grad)**2)
    diffcoeffgrad = th.sum( (coeff.grad-coefftarget.grad)**2)
    diffbiasgrad = th.sum( (bias.grad-biastarget.grad)**2)

    print("diffxgrad")
    print(diffxgrad)
    print( "diffcoeffgrad")
    print( diffcoeffgrad)
    print("diffbiasgrad")
    print(diffbiasgrad)


if __name__ == "__main__":
    demo()