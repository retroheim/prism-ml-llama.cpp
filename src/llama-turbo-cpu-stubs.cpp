// CPU-only weak stubs for TurboQuant InnerQ symbols.
// When ggml-cuda is built, the strong definitions in turbo-innerq.cu take
// precedence over these.

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

}
