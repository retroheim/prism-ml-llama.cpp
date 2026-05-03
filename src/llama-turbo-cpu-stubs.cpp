// CPU-only weak stubs for TurboQuant InnerQ + TriAttention GPU symbols.
// When ggml-cuda is built, the strong definitions in turbo-innerq.cu and
// triattention-score.cu take precedence over these.

#include "ggml.h"

#ifndef INNERQ_MAX_CHANNELS
#define INNERQ_MAX_CHANNELS 128
#endif

extern "C" {

__attribute__((weak)) bool  g_innerq_finalized = false;
__attribute__((weak)) float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS] = {0};

__attribute__((weak)) int turbo_innerq_needs_tensor_update(void) {
    return 0;
}

__attribute__((weak)) void turbo_innerq_mark_tensor_updated(void) {
}

struct triattention_gpu_state;

__attribute__((weak)) struct triattention_gpu_state * triattention_gpu_init(int, int, int, int) {
    return nullptr;
}

__attribute__((weak)) void triattention_gpu_free(struct triattention_gpu_state *) {
}

__attribute__((weak)) void triattention_gpu_free_dev(void *) {
}

__attribute__((weak)) void triattention_gpu_upload_cells(struct triattention_gpu_state *, const int *, int) {
}

__attribute__((weak)) float * triattention_gpu_alloc_scores(struct triattention_gpu_state *, int) {
    return nullptr;
}

__attribute__((weak)) void triattention_gpu_score_head(struct triattention_gpu_state *, const ggml_tensor *, int, float *, int, int, int, int, float, int) {
}

__attribute__((weak)) void triattention_gpu_scores_to_host(const float *, float *, int) {
}

}
