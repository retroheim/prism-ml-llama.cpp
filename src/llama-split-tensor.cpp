#include "llama-split-tensor.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <cmath>
#include <cstdio>
#include <string>

// Direct port of ik_llama's prepare_split_tensors (src/llama-load-tensors.cpp).
// See llama-split-tensor.h for contract.
void llama_prepare_split_tensors(
        int                          split_dim,
        ggml_context *               ctx,
        ggml_tensor *                tensor,
        llama_split_tensor &         out,
        const std::vector<int> &     splits,
        std::vector<size_t> &        mem_used) {
    GGML_ASSERT(split_dim <= 2);
    GGML_ASSERT(splits.size() > 1);
    GGML_ASSERT(mem_used.size() >= splits.size());

    const std::string base_name{tensor->name};
    out.tensor_splits.assign(splits.size(), nullptr);

    if (split_dim < 0) {
        // Replicate full tensor on every device (no actual splitting).
        for (size_t i = 0; i < splits.size(); ++i) {
            if (splits[i] > 0) {
                out.tensor_splits[i] = ggml_new_tensor_3d(ctx, tensor->type,
                                                         tensor->ne[0], tensor->ne[1], tensor->ne[2]);
                const std::string name_i = base_name + '.' + std::to_string(i);
                ggml_set_name(out.tensor_splits[i], name_i.c_str());
            }
        }
    } else if (split_dim == 0) {
        for (size_t i = 0; i < splits.size(); ++i) {
            if (splits[i] > 0) {
                out.tensor_splits[i] = ggml_new_tensor_3d(ctx, tensor->type,
                                                         splits[i], tensor->ne[1], tensor->ne[2]);
                const std::string name_i = base_name + '.' + std::to_string(i);
                ggml_set_name(out.tensor_splits[i], name_i.c_str());
            }
        }
    } else if (split_dim == 1) {
        for (size_t i = 0; i < splits.size(); ++i) {
            if (splits[i] > 0) {
                out.tensor_splits[i] = ggml_new_tensor_3d(ctx, tensor->type,
                                                         tensor->ne[0], splits[i], tensor->ne[2]);
                const std::string name_i = base_name + '.' + std::to_string(i);
                ggml_set_name(out.tensor_splits[i], name_i.c_str());
            }
        }
    } else {
        // split_dim == 2
        for (size_t i = 0; i < splits.size(); ++i) {
            if (splits[i] > 0) {
                out.tensor_splits[i] = ggml_new_tensor_3d(ctx, tensor->type,
                                                         tensor->ne[0], tensor->ne[1], splits[i]);
                const std::string name_i = base_name + '.' + std::to_string(i);
                ggml_set_name(out.tensor_splits[i], name_i.c_str());
            }
        }
    }

    out.ggml.n_device  = (int) splits.size();
    out.ggml.split_dim = split_dim;
    out.ggml.tensor    = tensor;
    out.ggml.splits    = out.tensor_splits.data();

    // Track memory consumed per device so callers can rebalance subsequent splits.
    for (int i = 0; i < out.ggml.n_device; ++i) {
        if (out.ggml.splits[i]) {
            mem_used[i] += ggml_nbytes(out.ggml.splits[i]);
        }
    }
}

// Direct port of ik_llama's create_split. Mem-balanced rounding of cumulative split fractions.
std::vector<int> llama_create_split_plan(
        int nr,
        int granularity,
        const std::vector<float> & splits,
        const std::vector<size_t> & mem_used,
        bool verbose) {
    GGML_ASSERT(!splits.empty());

    if (granularity < 0) {
        return std::vector<int>(splits.size(), nr);
    }

    GGML_ASSERT(nr % granularity == 0);
    GGML_ASSERT(mem_used.size() == splits.size());

    size_t tot_memory_used = 1;
    for (auto & mem : mem_used) {
        tot_memory_used += mem;
    }

    const int nchunk = nr / granularity;
    std::vector<int> result(splits.size());
    float last_split = 0.0f;
    int sum = 0;
    if (verbose) {
        LLAMA_LOG_INFO("--- %s: %d chunks\n", __func__, nchunk);
    }
    for (int i = 0; i < (int) splits.size(); ++i) {
        float p = splits[i] - last_split;
        const float p0 = p;
        p += (p - 1.0f * mem_used[i] / tot_memory_used);
        result[i] = (int) std::roundf(p * nchunk);
        if (result[i] < 0) {
            result[i] = 0;
        }
        if (verbose) {
            LLAMA_LOG_INFO("i = %d, p0 = %g, p = %g, result = %d\n", i, p0, p, result[i]);
        }
        sum += result[i];
        last_split = splits[i];
    }

    // Trim to total chunk count, preferring the device with greatest over-allocation.
    while (sum > nchunk) {
        last_split = 0.0f;
        float best_err = -INFINITY;
        int   ibest    = -1;
        for (int i = 0; i < (int) splits.size(); ++i) {
            if (result[i] > 0) {
                const float p      = splits[i] - last_split + (splits[i] - last_split - 1.0f * mem_used[i] / tot_memory_used);
                const float n_want = p * nchunk;
                const float err    = result[i] - n_want;
                if (err > best_err) {
                    best_err = err;
                    ibest    = i;
                }
            }
            last_split = splits[i];
        }
        GGML_ASSERT(ibest >= 0 && result[ibest] > 0);
        --result[ibest];
        --sum;
    }
    while (sum < nchunk) {
        last_split = 0.0f;
        float best_err = -INFINITY;
        int   ibest    = -1;
        for (int i = 0; i < (int) splits.size(); ++i) {
            const float p      = splits[i] - last_split + (splits[i] - last_split - 1.0f * mem_used[i] / tot_memory_used);
            const float n_want = p * nchunk;
            const float err    = n_want - result[i];
            if (err > best_err) {
                best_err = err;
                ibest    = i;
            }
            last_split = splits[i];
        }
        GGML_ASSERT(ibest >= 0);
        ++result[ibest];
        ++sum;
    }

    // Convert chunk counts back to element counts.
    for (auto & r : result) {
        r *= granularity;
    }
    return result;
}

// ik_llama port (split-mode-graph): post-load pass.
//
// Hooked from llama_model::load_tensors after done_getting_tensors() and before backend
// buffer allocation. Walks per-layer weights and (per arch) calls
// llama_prepare_split_tensors + model.register_split_graph_tensor.
//
// Today this is a guarded no-op:
//   - LLAMA_SPLIT_MODE_GRAPH is an explicit user opt-in
//   - n_devices > 1 prevents single-GPU configurations from going through the GRAPH path
//   - per-arch implementations are stubbed; expand them as each arch is brought online
//     with dual-GPU validation.
//
// The empty body is intentional: it makes the integration point explicit and lets the I/O
// dispatch (already wired in llama-context.cpp) remain dormant until per-arch surgery
// lands. No tensor is registered, so model.is_split_graph_tensor returns false everywhere
// and behaviour matches ROW mode.
void llama_split_graph_post_load_pass(
        llama_model &              model,
        const llama_model_params & params,
        ggml_context *             ctx_split) {
    if (params.split_mode != LLAMA_SPLIT_MODE_GRAPH) {
        return;
    }

    // Multi-GPU gate: count populated devices in params (NULL-terminated).
    int n_devices = 0;
    if (params.devices) {
        while (params.devices[n_devices]) {
            ++n_devices;
        }
    }
    if (n_devices < 2) {
        // Single-GPU GRAPH mode falls back to ROW behaviour with the existing CUDA split
        // buffer; nothing to do here.
        return;
    }

    GGML_UNUSED(ctx_split);
    GGML_UNUSED(model);

    LLAMA_LOG_INFO("%s: split-mode-graph post-load pass active for %d devices, but per-arch population is not yet implemented; falling back to ROW-equivalent behaviour\n",
                   __func__, n_devices);

    // Per-arch dispatch goes here, e.g.:
    //   switch (model.arch) {
    //       case LLM_ARCH_LLAMA:
    //           split_graph_prepare_llama(model, n_devices, ctx_split);
    //           break;
    //       case LLM_ARCH_QWEN35:
    //           split_graph_prepare_qwen35(model, n_devices, ctx_split);
    //           break;
    //       ...
    //   }
}
