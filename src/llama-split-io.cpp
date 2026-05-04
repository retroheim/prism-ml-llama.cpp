#include "llama-split-io.h"

#include "ggml-backend.h"

#include <cstring>

// Direct port of ik_llama's read_kv_cache_data_split (src/llama.cpp).
void llama_split_io_read_kv_cache_rows(
        ggml_tensor * tensor,
        const ggml_tensor * kv_for_dims,
        const uint8_t * data,
        size_t head,
        size_t row_size,
        int nrows) {
    auto extra    = (const ggml_split_tensor_t *) tensor->extra;
    auto kv_extra = kv_for_dims ? (const ggml_split_tensor_t *) kv_for_dims->extra : nullptr;
    GGML_ASSERT(extra && (kv_for_dims == nullptr || kv_extra));

    const int64_t ne_unified = kv_for_dims ? kv_for_dims->ne[1] : tensor->ne[0];
    GGML_ASSERT(row_size == ggml_row_size(tensor->type, ne_unified));

    size_t sum_ne             = 0;
    size_t sum_split_row_size = 0;
    std::vector<uint8_t> aux;

    for (int id = 0; id < extra->n_device; ++id) {
        ggml_tensor * split    = extra->splits[id];
        ggml_tensor * kv_split = kv_extra ? kv_extra->splits[id] : nullptr;
        GGML_ASSERT((split && (kv_split || kv_for_dims == nullptr)) || (!split && !kv_split));
        if (!split) {
            continue;
        }
        GGML_ASSERT(split->type == tensor->type);
        const int64_t ne_split       = kv_split ? kv_split->ne[1] : split->ne[0];
        const size_t  split_row_size = ggml_row_size(tensor->type, ne_split);
        aux.resize(split_row_size * (size_t) nrows);

        const uint8_t * src = data + sum_split_row_size;
        uint8_t * dst       = aux.data();
        for (int row = 0; row < nrows; ++row) {
            std::memcpy(dst, src, split_row_size);
            dst += split_row_size;
            src += row_size;
        }
        ggml_backend_tensor_set(split, aux.data(), head * split_row_size, (size_t) nrows * split_row_size);

        sum_ne             += ne_split;
        sum_split_row_size += split_row_size;
    }

    GGML_ASSERT(sum_ne == (size_t) ne_unified);
    GGML_ASSERT(sum_split_row_size == row_size);
}

// Direct port of ik_llama's get_tensor_data_split (KV variant).
void llama_split_io_write_kv_cache_rows(
        uint8_t * ptr,
        const ggml_tensor * tensor,
        const ggml_tensor * kv_for_dims,
        std::vector<uint8_t> & aux_buffer,
        size_t offset,
        size_t size) {
    GGML_ASSERT(kv_for_dims != nullptr);

    const int64_t ne_unified    = kv_for_dims->ne[1];
    const size_t  full_row_size = ggml_row_size(tensor->type, ne_unified);
    GGML_ASSERT(offset % full_row_size == 0);
    GGML_ASSERT(size   % full_row_size == 0);

    const size_t first_row = offset / full_row_size;
    const size_t num_rows  = size   / full_row_size;

    auto extra    = (const ggml_split_tensor_t *) tensor->extra;
    auto kv_extra = (const ggml_split_tensor_t *) kv_for_dims->extra;
    GGML_ASSERT(extra && kv_extra);

    size_t split_offset = 0;
    size_t total_size   = 0;

    for (int id = 0; id < extra->n_device; ++id) {
        ggml_tensor * split    = extra->splits[id];
        ggml_tensor * kv_split = kv_extra->splits[id];
        GGML_ASSERT((split && kv_split) || (!split && !kv_split));
        if (!split) {
            continue;
        }
        GGML_ASSERT(split->type == tensor->type);
        const size_t split_row_size = ggml_row_size(tensor->type, kv_split->ne[1]);
        const size_t split_size     = split_row_size * num_rows;
        if (split_size > aux_buffer.size()) {
            aux_buffer.resize(split_size);
        }
        ggml_backend_tensor_get(split, aux_buffer.data(), first_row * split_row_size, split_size);

        uint8_t * dst = ptr + split_offset;
        const uint8_t * src = aux_buffer.data();
        for (size_t row = 0; row < num_rows; ++row) {
            std::memcpy(dst, src, split_row_size);
            dst += full_row_size;
            src += split_row_size;
        }
        split_offset += split_row_size;
        total_size   += split_size;
    }

    GGML_ASSERT(total_size == size);
}

// Direct port of ik_llama's get_tensor_data_split (no-kv variant - e.g. recurrent state).
void llama_split_io_write_tensor_rows(
        uint8_t * ptr,
        const ggml_tensor * tensor,
        std::vector<uint8_t> & aux_buffer,
        size_t offset,
        size_t size) {
    const int64_t ne_unified    = tensor->ne[0];
    const size_t  full_row_size = ggml_row_size(tensor->type, ne_unified);
    GGML_ASSERT(offset % full_row_size == 0);
    GGML_ASSERT(size   % full_row_size == 0);

    const size_t first_row = offset / full_row_size;
    const size_t num_rows  = size   / full_row_size;

    auto extra = (const ggml_split_tensor_t *) tensor->extra;
    GGML_ASSERT(extra);

    size_t split_offset = 0;
    size_t total_size   = 0;

    for (int id = 0; id < extra->n_device; ++id) {
        ggml_tensor * split = extra->splits[id];
        if (!split) {
            continue;
        }
        GGML_ASSERT(split->type == tensor->type);
        const size_t split_row_size = ggml_row_size(tensor->type, split->ne[0]);
        const size_t split_size     = split_row_size * num_rows;
        if (split_size > aux_buffer.size()) {
            aux_buffer.resize(split_size);
        }
        ggml_backend_tensor_get(split, aux_buffer.data(), first_row * split_row_size, split_size);

        uint8_t * dst = ptr + split_offset;
        const uint8_t * src = aux_buffer.data();
        for (size_t row = 0; row < num_rows; ++row) {
            std::memcpy(dst, src, split_row_size);
            dst += full_row_size;
            src += split_row_size;
        }
        split_offset += split_row_size;
        total_size   += split_size;
    }

    GGML_ASSERT(total_size == size);
}
