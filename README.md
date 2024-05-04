# FusedFourierKAN
C++ and Cuda ops for fused FourierKAN. See https://github.com/GistNoesis/FourierKAN for naive version explaining what it's about (and be aware that dimensions order may differ and be subject to change). 

# LICENSE
Code is proprietary non-commercial, for research purposes only. 
Contact us gistnoesis@gmail.com for commercial licenses.
See LICENSE file for additional disclaimers.

# What is this about
Writing a custom op allow to not materialize memory. Zero extra memory needed. In addition it also allows to do some trigonometric trick to compute cos(k*x) and sin(k*x) more efficiently.

The core is quite simple : 
https://github.com/GistNoesis/FusedFourierKAN/blob/4bc5c3ae755ded9cea4c776ea8620fa3e435d2ae/ffkan.cpp#L6-L49

We had to write the forward and backward ops, for cpu and gpu, and some wrapper to make it available to pytorch.

In ffKANFunction.py and ffKANGPUFunction.py we verify that the functions and their gradient are approximately the same as the target.

The GPU version is not optimized, but run in parallel and is deterministic, in particular memory access are not yet coalesced or cached.

The structure and CMakeLists.txt allows for fast compilation time, that allows rapid iteration.

# INSTALL
Sorry it'll probably be painful. Only tested on linux

It still has rough edges but the happy path is the following : 
```
git clone https://github.com/GistNoesis/FusedFourierKAN.git
cd FusedFourierKAN
cd build
cmake ..
make
cd ..
pip install -e .
```

For it to be able to install properly you need to have nvcc working (and ideally of the same cuda version as the torch)
If cmake doesn't find torch, you should set TorchDir to the repository containing "TorchConfig.cmake" 
```locate TorchConfig.cmake```
```export Torch_DIR=folderContainingTorchConfig.cmake```
then ```cmake ..``` should work

# USAGE

Once install is done successfully should be smooth.
```from FusedFourierKAN.FusedFourierKANLayer import FusedFourierKANLayer```

You can also call demo function
```
from FusedFourierKAN.FusedFourierKANLayer import demo
demo()
```


