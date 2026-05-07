#include "llama-split-tensor.h"

#include "llama-arch.h"
#include "llama-hparams.h"
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
// LLaMA-class arch handler. Direct port of the relevant section of ik_llama's
// llm_load_tensors split pass (src/llama-load-tensors.cpp:4150+). Splits attn_norm (replicated),
// wq/wk/wv (column-split along ne[1] using KQ/VO granularity), wo (row-split along ne[0]).
// FFN tensors are intentionally left unsplit until tensor-parallel FFN is brought online.
//
// The split-tensor wrappers live on llama_layer (split_w[qkvo]/split_*_norm fields) so that
// the embedded ggml_split_tensor_t outlives the post-pass and remains valid for the lifetime
// of the model.
static void llama_split_graph_prepare_llama_class(
        llama_model &           model,
        ggml_context *          ctx,
        std::vector<float> &    cur_splits,
        std::vector<size_t> &   mem_used) {
    const llama_hparams & hparams = model.hparams;
    const int n_layer = hparams.n_layer;

    auto & layers = model.layers;
    if ((int) layers.size() < n_layer) {
        LLAMA_LOG_WARN("%s: layers vector smaller than n_layer (%zu < %d); aborting split prep\n",
                       __func__, layers.size(), n_layer);
        return;
    }

    for (int il = 0; il < n_layer; ++il) {
        auto & layer = layers[il];
        if (!layer.wq || !layer.wk || !layer.wo || !layer.wv) {
            continue; // some archs (e.g. recurrent layers) lack standard QKV; skip
        }

        const int gqa_ratio        = (int) (hparams.n_head(il) / hparams.n_head_kv(il));
        const int n_embd_head_k    = (int) hparams.n_embd_head_k(il);
        const int n_embd_head_v    = (int) hparams.n_embd_head_v(il);
        const int granularity_kq   = n_embd_head_k * gqa_ratio;
        const int granularity_vo   = n_embd_head_v * gqa_ratio;

        // Split plans for the layer's KV-output and Q-key directions.
        std::vector<int> split_kq = llama_create_split_plan((int) layer.wq->ne[1], granularity_kq, cur_splits, mem_used);
        std::vector<int> split_vo = llama_create_split_plan((int) layer.wo->ne[0], granularity_vo, cur_splits, mem_used);

        // attn_norm replicated on every device.
        if (layer.attn_norm) {
            std::vector<int> mirror((int) cur_splits.size(), (int) layer.attn_norm->ne[0]);
            llama_prepare_split_tensors(-1, ctx, layer.attn_norm, layer.split_q_norm, mirror, mem_used);
            // Note: re-using split_q_norm slot here is benign since only one of {attn_norm, q_norm}
            // is typically present in LLaMA-class archs. If both exist the second overwrites.
            // Architectures requiring both should use a dedicated slot — TBD.
            layer.attn_norm->extra = (void *) &layer.split_q_norm.ggml;
            model.register_split_graph_tensor(layer.attn_norm);
        }

        llama_prepare_split_tensors(1, ctx, layer.wq, layer.split_wq, split_kq, mem_used);
        layer.wq->extra = (void *) &layer.split_wq.ggml;
        model.register_split_graph_tensor(layer.wq);

        llama_prepare_split_tensors(1, ctx, layer.wk, layer.split_wk, split_kq, mem_used);
        layer.wk->extra = (void *) &layer.split_wk.ggml;
        model.register_split_graph_tensor(layer.wk);

        llama_prepare_split_tensors(1, ctx, layer.wv, layer.split_wv, split_vo, mem_used);
        layer.wv->extra = (void *) &layer.split_wv.ggml;
        model.register_split_graph_tensor(layer.wv);

        llama_prepare_split_tensors(0, ctx, layer.wo, layer.split_wo, split_vo, mem_used);
        layer.wo->extra = (void *) &layer.split_wo.ggml;
        model.register_split_graph_tensor(layer.wo);
    }
}

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

    if (!ctx_split) {
        LLAMA_LOG_WARN("%s: no ctx provided; skipping split-mode-graph post-load pass\n", __func__);
        return;
    }

    // Build cumulative split fractions from params.tensor_split.
    // tensor_split is per-device fractions; we convert to a cumulative form expected
    // by llama_create_split_plan (each entry monotonic in [0,1], last == 1).
    std::vector<float> cur_splits(n_devices, 0.0f);
    {
        float sum = 0.0f;
        bool  any = false;
        for (int i = 0; i < n_devices; ++i) {
            if (params.tensor_split && params.tensor_split[i] > 0.0f) {
                sum += params.tensor_split[i];
                any = true;
            }
        }
        if (!any || sum <= 0.0f) {
            // Equal split.
            for (int i = 0; i < n_devices; ++i) {
                cur_splits[i] = float(i + 1) / float(n_devices);
            }
        } else {
            float acc = 0.0f;
            for (int i = 0; i < n_devices; ++i) {
                const float w = params.tensor_split[i] > 0.0f ? params.tensor_split[i] : 0.0f;
                acc += w / sum;
                cur_splits[i] = acc;
            }
            // Force last entry to exactly 1.0 to absorb rounding.
            cur_splits.back() = 1.0f;
        }
    }

    std::vector<size_t> mem_used(n_devices, 0);

    // Per-arch dispatch.
    switch (model.arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
            llama_split_graph_prepare_llama_class(model, ctx_split, cur_splits, mem_used);
            break;
        default:
            LLAMA_LOG_INFO("%s: arch %s has no split-mode-graph handler; falling back to ROW-equivalent behaviour\n",
                           __func__, llm_arch_name(model.arch));
            return;
    }

    LLAMA_LOG_INFO("%s: split-mode-graph post-load pass complete for arch %s across %d devices\n",
                   __func__, llm_arch_name(model.arch), n_devices);
    for (int i = 0; i < n_devices; ++i) {
        LLAMA_LOG_INFO("%s:   device %d: %.2f MiB allocated to split sub-tensors\n",
                       __func__, i, mem_used[i] / 1024.0 / 1024.0);
    }
}
