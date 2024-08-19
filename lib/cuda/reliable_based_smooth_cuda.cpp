#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

void reliable_based_smooth_cuda(torch::Tensor reliable_area, torch::Tensor param, torch::Tensor grad, float wx, float wy, float wz, bool dense_mode);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

void reliable_based_smooth(torch::Tensor reliable_area, torch::Tensor param, torch::Tensor grad, float wx, float wy, float wz, bool dense_mode)
{
    CHECK_INPUT(reliable_area);
    CHECK_INPUT(param);
    CHECK_INPUT(grad);
    reliable_based_smooth_cuda(reliable_area, param, grad, wx, wy, wz, dense_mode);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("reliable_based_smooth", &reliable_based_smooth, "Add reliable based smooth");
}
