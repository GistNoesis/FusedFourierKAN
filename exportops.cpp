#include <torch/torch.h>
#include <vector>

#include "op.h"
#include "opGPU.h"


TORCH_LIBRARY(KAN_ops, m) {
  m.def("ffKANForward", ffKANForward);
  m.def("ffKANBackward", ffKANBackward);
  m.def("ffKANGPUForward", ffKANGPUForward);
  m.def("ffKANGPUBackward", ffKANGPUBackward);
}
