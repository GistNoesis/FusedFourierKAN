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
Sorry it'll probably be painful. Only tested on linux, (although some users got it to work on windows with minor modifications to the loading path see https://github.com/GistNoesis/FusedFourierKAN/issues/3 )

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

Some users are encountering some build errors with recent g++ version. using g++-10 was suggested. (See https://github.com/GistNoesis/FusedFourierKAN/issues/1 )

(My current versions, where it compiles and run fine are : 
```
g++ --version
g++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
```
```
nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Mon_Apr__3_17:16:06_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
Build cuda_12.1.r12.1/compiler.32688072_0
```
```
import torch as th
>>> th.__version__
'2.2.1+cu121'
```
)

# USAGE

Once install is done successfully should be smooth.
```from FusedFourierKAN.FusedFourierKANLayer import FusedFourierKANLayer```

You can also call demo function
```
from FusedFourierKAN.FusedFourierKANLayer import demo
demo()
```


