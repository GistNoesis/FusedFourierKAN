import torch as th
import numpy as np

from .ffKANFunction import ffKANFunction
from .ffKANGPUFunction import ffKANGPUFunction

#This is inspired by Kolmogorov-Arnold Networks but using 1d fourier coefficients instead of splines coefficients
#It should be easier to optimize as fourier are more dense than spline (global vs local)
#Once convergence is reached you can replace the 1d function with spline approximation for faster evaluation giving almost the same result
#The other advantage of using fourier over spline is that the function are periodic, and therefore more numerically bounded
#Avoiding the issues of going out of grid

class FusedFourierKANLayer(th.nn.Module):
    def __init__( self, inputdim, outdim, gridsize,addbias=True):
        super(FusedFourierKANLayer,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        
        #The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        #then each coordinates of the output is of unit variance 
        #independently of the various sizes
        self.fouriercoeffs = th.nn.Parameter( th.randn(2,inputdim,outdim,gridsize) / 
                                             (np.sqrt(inputdim) * np.sqrt(self.gridsize) ) )
        if( self.addbias ):
            self.bias  = th.nn.Parameter( th.zeros(outdim))

    #x.shape ( ... , indim ) 
    #out.shape ( ..., outdim)
    def forward(self,x):
        xshp = x.shape
        outshape = xshp[0:-1]+(self.outdim,)
        x = th.reshape(x,(-1,self.inputdim))

        if x.get_device() == -1:
            #print("cpu")
            y = ffKANFunction(x,self.fouriercoeffs,self.bias)
        else:
            #print("not cpu")
            y = ffKANGPUFunction(x,self.fouriercoeffs,self.bias)
        y = th.reshape( y, outshape)
        return y

def demo():
    bs = 10
    L = 3 #Not necessary just to show that additional dimensions are batched like Linear
    inputdim = 5
    hidden = 20
    outdim = 10
    gridsize = 30

    device = "cuda" #cpu

    fkan1 = FusedFourierKANLayer(inputdim, hidden, gridsize).to(device)
    fkan2 = FusedFourierKANLayer(hidden, outdim, gridsize).to(device)

    x0 =th.randn(bs,inputdim).to(device)

    h = fkan1(x0)
    y = fkan2(h)
    print("x0.shape")
    print( x0.shape)
    print("h.shape")
    print( h.shape)
    print( "th.mean( h )")
    print( th.mean( h ) )
    print( "th.mean( th.var(h,-1) )")
    print( th.mean( th.var(h,-1)))

    print("y.shape")
    print( y.shape )
    print( "th.mean( y)")
    print( th.mean( y ) )
    print( "th.mean( th.var(y,-1) )")
    print( th.mean( th.var(y,-1)))

    print(" ")
    print(" ")
    print("Sequence example")
    print(" ")
    print(" ")
    xseq =th.randn(bs, L ,inputdim).to(device)

    h = fkan1(xseq)
    y = fkan2(h)
    print("xseq.shape")
    print( xseq.shape)
    print("h.shape")
    print( h.shape)
    print( "th.mean( h )")
    print( th.mean( h ) )
    print( "th.mean( th.var(h,-1) )")
    print( th.mean( th.var(h,-1)))

    print("y.shape")
    print( y.shape )
    print( "th.mean( y)")
    print( th.mean( y ) )
    print( "th.mean( th.var(y,-1) )")
    print( th.mean( th.var(y,-1)))

if __name__ == "__main__":
    demo()