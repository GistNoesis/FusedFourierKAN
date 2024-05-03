cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(fusedFourierKAN)

find_package(Torch REQUIRED)
find_package(CUDA QUIET REQUIRED)

# Define our library target
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};-O3 -Xcompiler -fPIC
    )
set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_library( fusedFourierKANGPU 
                  kernels/fusedFourierKAN.cu 
                  )

add_library(fusedFourierKAN SHARED op.cpp opGPU.cpp ffkan.cpp exportops.cpp)
# Enable C++17
target_compile_features(fusedFourierKAN PRIVATE cxx_std_17)
# Link against LibTorch
target_link_libraries(fusedFourierKAN "${TORCH_LIBRARIES}" fusedFourierKANGPU)
