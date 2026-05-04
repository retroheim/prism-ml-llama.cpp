#pragma once

#include "common.cuh"

// ik_llama port (split-mode-graph): n-ary reduction across same-shape source tensors.
// Op is GGML_OP_ADD or GGML_OP_MEAN. Used at the output of tensor-parallel attention to
// gather per-device partial results back into a single F32 output.
void ggml_cuda_op_reduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
