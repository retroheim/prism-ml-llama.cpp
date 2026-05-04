#pragma once

// ik_llama port: split-mode-graph helpers
//
// LLAMA_SPLIT_MODE_GRAPH is a multi-GPU split strategy ported from ik_llama.
// It evolves the existing ROW path with:
//   - cross-GPU KV cache sync for hybrid CPU+GPU offload
//   - smarter scheduling under tensor overrides (-smgs)
//   - max-GPU cap for asymmetric multi-GPU setups
//   - auto-disable CUDA graphs (CUDA graph capture conflicts with cross-GPU sync)
//
// This module isolates graph-mode decisions so prism's refactored llama-context /
// llama-graph / llama-model-loader files only need minimal hook points.
//
// Current state:
//
//   Phase 1 (CLI + cparams)                                : DONE
//   Phase 2 (backend sched stub)                           : DONE
//   Phase 3a (row-aligned partial I/O on CUDA split buffer): DONE
//   Phase 3.1 (ggml_split_tensor_t)                        : DONE  ggml/include/ggml.h
//   Phase 3.2 (GGML_OP_REDUCE + GGML_OP_FAKE_CPY)          : DONE  CPU + CUDA dispatch
//   Phase 3.3 (extra struct + helpers ported from ik)      : DONE  llama-split-tensor.{h,cpp}
//   Phase 3.4 (storage in llama_layer + llama_kv_cache)    : DONE  fields added
//   Phase 3.4 (model-loader post-pass + KV alloc hookup)   : TBD   needs per-arch surgery
//   Phase 3.5 (tensor-parallel attention via build_attn)   : TBD   most invasive, needs 2-GPU
//   Phase 3.6 (split-aware session save/load I/O ports)    : DONE  llama-split-io.{h,cpp}
//   Phase 3.6 (dispatch from llama_io_read/write_tensor)   : TBD   one-line wiring
//   Phase 4   (CUDA-graph auto-disable)                    : DONE  prism existing check covers
//   Phase 5   (Qwen3.5-MoE/Qwen3-Next hybrid fix)          : TBD   depends on 3.5
//
// Direct-port status: every helper from ik_llama PR #1048 (read_kv_cache_data_split,
// get_tensor_data_split) and from the PR introducing split tensors (prepare_split_tensors,
// create_split, ggml_split_tensor_t) is ported verbatim and compiles. The remaining work is
// integration: walking each model arch to call prepare_split_tensors on its weights,
// rewriting build_attn for tensor-parallel paths, and dispatching read/write_tensor to the
// split-io helpers when extra type is ggml_split_tensor_t.
//
// These integration steps are correctness-critical and span ~70 model files. Without a
// dual-GPU testbed, integration must be staged arch-by-arch with smoke tests; this is left
// to the next session against actual 2-GPU hardware.
//
// Architectural notes (carried forward):
//   - ik's GRAPH mode splits weights along ne[1] (n_embd_k_gqa output dim) and KV along
//     ne[0] (head dim, with sub-tensor ne derived from weights). prism's CUDA split buffer
//     splits along ggml_nrows. For weights ggml_nrows == ne[1] (matches ik); for KV cache
//     prism's path would split tokens (wrong), which is why ik allocates KV sub-tensors
//     manually rather than using the split buffer. Phase 3.4 KV hookup must follow the same
//     pattern: allocate sub-tensors via ggml_new_tensor_*d in the regular CUDA buft for
//     each device, store them in llama_split_tensor.tensor_splits, and assign to extra.

#include "llama.h"

#include <cstdint>

namespace llama_split_graph {

// True when split mode wants per-row tensor splitting across GPUs.
// Both ROW (mainline tensor parallelism) and GRAPH (ik graph mode) qualify.
inline bool wants_split_buffer(llama_split_mode mode) {
    return mode == LLAMA_SPLIT_MODE_ROW || mode == LLAMA_SPLIT_MODE_GRAPH;
}

// True when split mode is graph mode (ik port).
inline bool is_graph_mode(llama_split_mode mode) {
    return mode == LLAMA_SPLIT_MODE_GRAPH;
}

// True when CUDA graph capture should be disabled. Graph mode performs cross-GPU
// KV sync at decode-time, which is incompatible with stream capture replay.
// Override-friendly: respects existing GGML_CUDA_DISABLE_GRAPHS env var.
inline bool should_disable_cuda_graphs(llama_split_mode mode) {
    return mode == LLAMA_SPLIT_MODE_GRAPH;
}

// Cap the GPU count used by graph mode. Returns the effective device count to use.
// `requested == 0` means "use all". `device_count` is what the backend has discovered.
inline uint32_t effective_gpu_count(int32_t requested, uint32_t device_count) {
    if (requested <= 0) {
        return device_count;
    }
    return (uint32_t) requested < device_count ? (uint32_t) requested : device_count;
}

} // namespace llama_split_graph
