#pragma once

// ik_llama port (split-mode-graph): C++ wrapper around ggml_split_tensor_t.
// Mirrors ik's `llama_split_tensor` struct in src/llama-load-tensors.cpp:
//   - owns the per-device sub-tensor pointer storage
//   - the embedded `ggml` member is what tensor->extra is set to
//
// Lifetime: instances are owned by the model (or KV cache) so that the embedded
// ggml_split_tensor_t and its splits[] array remain valid as long as the parent
// tensor is alive. Caller is responsible for placing the wrapper in a stable
// container (vector, deque, or model-owned arena) before assigning tensor->extra.

#include "ggml.h"

#include <cstddef>
#include <vector>

struct llama_split_tensor {
    // The C struct exposed via tensor->extra. Must outlive the parent tensor.
    ggml_split_tensor_t ggml = {0, 0, nullptr, nullptr};

    // Per-device sub-tensors. ggml.splits points at this vector's data.
    // splits.size() == ggml.n_device. Entries may be nullptr for devices that hold no slice.
    std::vector<ggml_tensor *> tensor_splits;
};

// Direct port of ik_llama's prepare_split_tensors (src/llama-load-tensors.cpp).
//
// Builds per-device sub-tensors of `tensor` along `split_dim`, allocating each from `ctx`.
// `splits[i]` is this device's slice size along split_dim (in elements). Entries that are
// zero produce a NULL sub-tensor entry. `mem_used[i]` is incremented by the byte size of
// the device-i slice, so callers can balance subsequent splits using the running memory
// budget per device.
//
// Constraints (mirroring ik):
//   - split_dim must be -1 (replicate full tensor on each device), 0, 1, or 2
//   - splits.size() > 1 (at least two devices)
//   - tensor's first three dims are honoured; higher-rank tensors are not yet supported
//
// On return, `out.ggml` is fully populated and ready to be assigned to tensor->extra.
void llama_prepare_split_tensors(
        int                            split_dim,
        struct ggml_context *          ctx,
        struct ggml_tensor *           tensor,
        struct llama_split_tensor &    out,
        const std::vector<int> &       splits,
        std::vector<size_t> &          mem_used);

// Direct port of ik_llama's create_split (src/llama-load-tensors.cpp).
//
// Computes per-device chunk counts for splitting `nr` rows across len(splits) devices.
// `granularity` is the minimum atomic chunk size (e.g. block size for quantised types,
// or head_dim when splitting a Q/K/V projection). `splits` is a cumulative split plan
// (each entry in [0,1], monotonically increasing, last == 1.0). `mem_used` is the
// running per-device byte budget used to balance subsequent splits.
//
// Returns a vector of length splits.size() with the chunk count assigned to each
// device. Sum of returned counts equals nr/granularity.
//
// `granularity < 0` is a special "replicate" plan: every device gets the full nr.
std::vector<int> llama_create_split_plan(
        int                            nr,
        int                            granularity,
        const std::vector<float> &     splits,
        const std::vector<size_t> &    mem_used,
        bool                           verbose = false);

// Forward declaration — defined in llama-model.h.
struct llama_model;
struct llama_model_params;

// ik_llama port (split-mode-graph): post-load pass that consumes per-layer weight tensors
// and (when split_mode == LLAMA_SPLIT_MODE_GRAPH and multi-GPU) prepares the per-device
// sub-tensors. Must be called between ml.done_getting_tensors() and backend buffer
// allocation in llama_model::load_tensors.
//
// On exit, every layer weight that was processed has:
//   - its tensor->extra set to a ggml_split_tensor_t pointer (owned by llama_layer)
//   - itself registered in model's split_graph_tensors set
//
// Today the implementation early-returns on any split mode other than GRAPH and on
// single-GPU configurations. Per-arch dispatch is stubbed; expand it in
// llama-split-tensor.cpp as each architecture is brought online with dual-GPU testing.
void llama_split_graph_post_load_pass(
        struct llama_model &              model,
        const struct llama_model_params & params,
        struct ggml_context *             ctx_split);
