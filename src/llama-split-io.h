#pragma once

// ik_llama port (split-mode-graph): split-aware tensor I/O for session save/load.
//
// When a KV cache tensor is split across GPUs (its extra is a ggml_split_tensor_t,
// not the standard ggml_tensor_extra_gpu), the row layout in serialized state
// differs from the in-memory layout: the host buffer stores rows as if the tensor
// were unified, but each device only holds a column slice along the split dim.
//
// These helpers reassemble (write) or distribute (read) per-device slices around
// the unified host buffer. Direct port of ik's read_kv_cache_data_split and
// get_tensor_data_split (src/llama.cpp #1048).

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Read `nrows` rows of `tensor` from the host buffer `data`, slicing per-device and
// dispatching to each split sub-tensor. `kv_for_dims` is the wk/wv weight whose
// per-device ne[1] determines the column-slice width per device (use the parent KV
// cache layer's wk for K cache, wv for V cache). Pass nullptr for non-KV tensors
// (e.g. recurrent state) where the split sub-tensor's own ne[0] gives the slice width.
//
// `head` is the starting row offset within each device's sub-tensor (in rows).
// `row_size` is bytes per logical (unsplit) row.
//
// Asserts that the extras are `ggml_split_tensor_t *` populated by the model loader.
void llama_split_io_read_kv_cache_rows(
        struct ggml_tensor *  tensor,
        const struct ggml_tensor * kv_for_dims,
        const uint8_t *       data,
        size_t                head,
        size_t                row_size,
        int                   nrows);

// Write `size` bytes of `tensor` (which must be split) into `ptr` (host buffer
// at row-aligned offset `offset`). Assembles the unified row layout by copying
// each device's slice into the right column band of each row.
//
// Variant with `kv_for_dims`: per-device column width = kv_for_dims->extra->splits[id]->ne[1].
// Variant without: per-device column width = tensor->extra->splits[id]->ne[0].
//
// `aux_buffer` is a reusable scratch buffer; will be grown as needed.
void llama_split_io_write_kv_cache_rows(
        uint8_t *                          ptr,
        const struct ggml_tensor *         tensor,
        const struct ggml_tensor *         kv_for_dims,
        std::vector<uint8_t> &             aux_buffer,
        size_t                             offset,
        size_t                             size);

void llama_split_io_write_tensor_rows(
        uint8_t *                          ptr,
        const struct ggml_tensor *         tensor,
        std::vector<uint8_t> &             aux_buffer,
        size_t                             offset,
        size_t                             size);
