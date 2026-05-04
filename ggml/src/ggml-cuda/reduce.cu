#include "reduce.cuh"

// ik_llama port (split-mode-graph): n-ary same-shape reduction.
// Each thread handles a strided range of elements. Op is GGML_OP_ADD or GGML_OP_MEAN
// (read from dst->op_params[0]); n source count is dst->op_params[1].
//
// Sources are expected to be F32 and contiguous-equivalent (same shape implies
// same layout when allocated together). Operates element-wise over ggml_nelements(dst).

template <int N>
static __global__ void k_reduce_add_n(const float ** __restrict__ srcs,
                                      float * __restrict__ dst,
                                      int64_t nelems,
                                      float scale) {
    const int64_t tid    = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;

    for (int64_t i = tid; i < nelems; i += stride) {
        float v = srcs[0][i];
        #pragma unroll
        for (int k = 1; k < N; ++k) {
            v += srcs[k][i];
        }
        dst[i] = v * scale;
    }
}

static __global__ void k_reduce_add(const float * const * __restrict__ srcs,
                                    int n,
                                    float * __restrict__ dst,
                                    int64_t nelems,
                                    float scale) {
    const int64_t tid    = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;

    for (int64_t i = tid; i < nelems; i += stride) {
        float v = srcs[0][i];
        for (int k = 1; k < n; ++k) {
            v += srcs[k][i];
        }
        dst[i] = v * scale;
    }
}

void ggml_cuda_op_reduce(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const enum ggml_op op = (enum ggml_op) ggml_get_op_params_i32(dst, 0);
    const int          n  = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(n >= 1 && n <= GGML_MAX_SRC);
    GGML_ASSERT(op == GGML_OP_ADD || op == GGML_OP_MEAN);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->src[0] != nullptr && dst->src[0]->type == GGML_TYPE_F32);

    const int64_t nelems = ggml_nelements(dst);
    const float   scale  = (op == GGML_OP_MEAN && n > 0) ? (1.0f / (float) n) : 1.0f;

    cudaStream_t stream = ctx.stream();
    float * dst_data = (float *) dst->data;

    // Stage source pointers in pinned host memory then copy to device — small payload.
    const float * h_srcs[GGML_MAX_SRC];
    for (int k = 0; k < n; ++k) {
        GGML_ASSERT(dst->src[k] != nullptr);
        GGML_ASSERT(ggml_are_same_shape(dst, dst->src[k]));
        h_srcs[k] = (const float *) dst->src[k]->data;
    }

    const int  block = 256;
    const int  grid  = std::min<int64_t>((nelems + block - 1) / block, 8192);

    if (n == 1) {
        // pass-through with optional mean scaling
        CUDA_CHECK(cudaMemcpyAsync(dst_data, h_srcs[0], (size_t) nelems * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
        return;
    }

    // Allocate device-side pointer table.
    ggml_cuda_pool_alloc<const float *> dev_srcs_alloc(ctx.pool(), n);
    const float ** d_srcs = dev_srcs_alloc.get();
    CUDA_CHECK(cudaMemcpyAsync(d_srcs, h_srcs, n * sizeof(const float *),
                               cudaMemcpyHostToDevice, stream));

    k_reduce_add<<<grid, block, 0, stream>>>(d_srcs, n, dst_data, nelems, scale);
}
