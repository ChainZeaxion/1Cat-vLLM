// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <torch/types.h>
#include "sm70_router.cuh"

namespace {
void check_output(const torch::Tensor& tensor, const torch::Tensor& logits,
                  at::ScalarType type) {
  TORCH_CHECK(tensor.device() == logits.device() &&
                  tensor.scalar_type() == type && tensor.is_contiguous() &&
                  tensor.sizes() == at::IntArrayRef({logits.size(0), 10}),
              "Expected same-device contiguous router output [M,10]");
}
void check_logits(const torch::Tensor& logits) {
  TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::kHalf &&
                  logits.is_contiguous() && logits.dim() == 2 &&
                  logits.size(0) >= 1 && logits.size(0) <= 20 &&
                  logits.size(1) == 512,
              "Expected contiguous CUDA FP16 router logits [M1..20,512]");
}
void check_sm70() {
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(properties->major == 7 && properties->minor == 0,
              "SM70 router requires V100");
}
void top10(torch::Tensor weights, torch::Tensor ids, torch::Tensor source,
           torch::Tensor logits) {
  check_logits(logits);
  const c10::cuda::CUDAGuard guard(logits.device());
  check_sm70();
  check_output(weights, logits, at::kFloat);
  check_output(ids, logits, at::kInt);
  check_output(source, logits, at::kInt);
  vllm::sm70_router::select_kernel<<<logits.size(0), 32, 0,
                                     at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(logits.data_ptr()),
      weights.data_ptr<float>(), ids.data_ptr<int>(), source.data_ptr<int>(),
      logits.size(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void select_quantize(torch::Tensor weights, torch::Tensor ids,
                     torch::Tensor source, torch::Tensor q8,
                     torch::Tensor logits, torch::Tensor x) {
  check_logits(logits);
  const c10::cuda::CUDAGuard guard(logits.device());
  check_sm70();
  check_output(weights, logits, at::kFloat);
  check_output(ids, logits, at::kInt);
  check_output(source, logits, at::kInt);
  const int m = logits.size(0);
  TORCH_CHECK(x.device() == logits.device() && x.scalar_type() == at::kHalf &&
                  x.is_contiguous() && x.sizes() == at::IntArrayRef({m, 2560}),
              "Expected FP16 router input [M,2560]");
  TORCH_CHECK(q8.device() == logits.device() && q8.scalar_type() == at::kByte &&
                  q8.is_contiguous() &&
                  q8.sizes() == at::IntArrayRef({m, 80, 36}),
              "Expected Q8_1 output [M,80,36]");
  vllm::sm70_router::select_quantize_kernel<<<
      dim3(10, m), 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const half*>(logits.data_ptr()),
      reinterpret_cast<const half*>(x.data_ptr()), weights.data_ptr<float>(),
      ids.data_ptr<int>(), source.data_ptr<int>(),
      reinterpret_cast<vllm::sm70_gguf::Q8_1*>(q8.data_ptr()), m);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace
// Note: v153 builds this file as a standalone `_sm70_router_C` module, so it
// carried a TORCH_LIBRARY(vllm_sm70_router) block and a PYBIND11_MODULE stub.
// Here it is compiled into `_C`; the exported op (sm70_dmv13_out) is registered
// in torch_bindings.cpp, and a PYBIND11_MODULE would clash with PyInit__C.
