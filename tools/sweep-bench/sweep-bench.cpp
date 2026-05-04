// sweep-bench: sweep prompt-processing and token-generation throughput
// across rising n_kv (context fill) at fixed n_ubatch.
// Ported from ik_llama.cpp (PR #1454, #1468, etc.) — adapted to prism API.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct sweep_opts {
    int  nrep         = 1;
    bool minilog      = false;
    bool jsonl        = false;
    bool batch_warmup = false;
};

// Strip sweep-bench-specific flags from argv so common_params_parse does not see them.
// Returns the remaining argc; argv is rewritten in place.
static int extract_sweep_opts(int argc, char ** argv, sweep_opts & out) {
    int w = 1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-r" || a == "--repetitions" || a == "-nrep") {
            if (i + 1 < argc) {
                out.nrep = std::atoi(argv[++i]);
                if (out.nrep < 1) out.nrep = 1;
            }
        } else if (a == "-minilog" || a == "--minilog") {
            out.minilog = true;
        } else if (a == "-jsonl" || a == "--output-jsonl") {
            out.jsonl = true;
        } else if (a == "-bwu" || a == "--batch-warmup") {
            out.batch_warmup = true;
        } else {
            argv[w++] = argv[i];
        }
    }
    return w;
}

static void llama_selective_log_callback(ggml_log_level level, const char * text, void * /*user_data*/) {
    (void) level;
    static const char * skip_patterns[] = {
        "Setting default device in layer",
        "llama_model_loader: Dumping metadata",
        "llama_model_loader: - kv  ",
        "llama_model_loader: - type ",
        "validate_override:",
        "load: printing all EOG",
        "load:   - ",
        "load: special tokens cache",
        "load: token to piece cache",
        "llm_load_print_meta:",
        "print_info:",
        "------------------- Layer sizes",
        "Layer ",
        "llm_load_tensors:",
        "==========================",
        "merging up/gate in layer",
        "repacking up/gate experts weight in layer",
    };
    for (const char * pat : skip_patterns) {
        if (strstr(text, pat) != nullptr) {
            return;
        }
    }
    int i = 0;
    while (text[i] == ' ' || text[i] == '\t') i++;
    if (text[i] == ',' || text[i] == '(' || text[i] == ')' || (text[i] >= '0' && text[i] <= '9')) {
        return;
    }
    LOG("%s", text);
}

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m model.gguf -c 8192 -b 2048 -ub 512\n", argv[0]);
    LOG("\nsweep-bench specific flags (parsed before common args):\n");
    LOG("    -r, --repetitions N      number of repetitions per measurement (default: 1)\n");
    LOG("    --minilog                suppress noisy model-loader output\n");
    LOG("    --output-jsonl           emit one JSON object per measurement\n");
    LOG("    --batch-warmup           run a full ubatch warmup pass before measuring\n");
    LOG("\n");
}

int main(int argc, char ** argv) {
    sweep_opts sopts;
    argc = extract_sweep_opts(argc, argv, sopts);

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_BENCH, print_usage)) {
        return 1;
    }

    if (sopts.minilog) {
        llama_log_set(llama_selective_log_callback, nullptr);
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params mparams = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == nullptr) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    llama_context_params cparams = common_context_params_to_llama(params);
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        LOG_ERR("%s: failed to create the llama_context\n", __func__);
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_token bos     = llama_vocab_bos(vocab);
    const int32_t     n_vocab = llama_vocab_n_tokens(vocab);

    const uint32_t n_kv_max = llama_n_ctx(ctx);
    llama_memory_t memory   = llama_get_memory(ctx);

    auto decode_helper = [](llama_context * ctx, llama_batch & batch, int32_t n_batch) {
        for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch) {
            const int32_t n_tokens = std::min(n_batch, (int32_t)(batch.n_tokens - i));
            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };
            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                LOG_ERR("failed to decode batch, n_batch=%d ret=%d\n", n_batch, ret);
                return false;
            }
            llama_synchronize(ctx);
        }
        return true;
    };

    const uint32_t pp = params.n_ubatch;
    const uint32_t tg = params.n_predict > 0 ? (uint32_t)params.n_predict : params.n_ubatch / 4;

    if (!sopts.jsonl) {
        LOG("\n");
        LOG("%s: n_kv_max=%u n_batch=%u n_ubatch=%u flash_attn=%d n_gpu_layers=%d n_threads=%d n_threads_batch=%d\n",
            __func__, n_kv_max, params.n_batch, params.n_ubatch,
            (int)params.flash_attn_type, params.n_gpu_layers,
            cparams.n_threads, cparams.n_threads_batch);
        LOG("\n");
        LOG("|%6s | %6s | %6s | %8s | %8s | %8s | %8s |\n", "PP", "TG", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s");
        LOG("|%6s-|-%6s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|\n", "------", "------", "------", "--------", "--------", "--------", "--------");
    }

    llama_batch batch = llama_batch_init(n_kv_max, 0, 1);

    if (params.warmup) {
        common_batch_add(batch, bos, 0, { 0 }, false);
        if (!decode_helper(ctx, batch, cparams.n_batch)) return 1;
    }
    if (sopts.batch_warmup) {
        llama_memory_seq_rm(memory, 0, params.n_ubatch, -1);
        common_batch_clear(batch);
        for (int32_t i = 0; i < params.n_ubatch; ++i) {
            common_batch_add(batch, std::rand() % n_vocab, i, { 0 }, false);
        }
        if (!decode_helper(ctx, batch, cparams.n_ubatch)) return 1;
    }

    common_batch_clear(batch);
    llama_memory_clear(memory, true);
    llama_perf_context_reset(ctx);

    int i_loop = 0;
    for (uint32_t n_kv = 0; n_kv < n_kv_max; n_kv += params.n_ubatch) {
        const int nrep = i_loop < 1 ? sopts.nrep : 1;

        // token generation pass
        const int64_t t_tg_start = ggml_time_us();
        for (int irep = 0; irep < nrep; ++irep) {
            llama_memory_seq_rm(memory, 0, n_kv, -1);
            for (uint32_t i = 0; i < tg; ++i) {
                common_batch_clear(batch);
                common_batch_add(batch, std::rand() % n_vocab, n_kv + i, { 0 }, true);
                if (!decode_helper(ctx, batch, cparams.n_batch)) return 1;
            }
        }
        const int64_t t_tg_end = ggml_time_us();

        // prompt processing pass
        const int64_t t_pp_start = ggml_time_us();
        for (int irep = 0; irep < nrep; ++irep) {
            llama_memory_seq_rm(memory, 0, n_kv, -1);
            common_batch_clear(batch);
            for (uint32_t i = 0; i < pp; ++i) {
                common_batch_add(batch, std::rand() % n_vocab, n_kv + i, { 0 }, false);
            }
            batch.logits[batch.n_tokens - 1] = true;
            if (!decode_helper(ctx, batch, cparams.n_batch)) return 1;
        }
        const int64_t t_pp_end = ggml_time_us();

        const float t_pp     = (t_pp_end - t_pp_start) / 1e6f / nrep;
        const float t_tg     = (t_tg_end - t_tg_start) / 1e6f / nrep;
        const float speed_pp = pp / t_pp;
        const float speed_tg = tg / t_tg;

        if (sopts.jsonl) {
            LOG("{\"n_kv_max\": %u, \"n_batch\": %u, \"n_ubatch\": %u, \"flash_attn\": %d, \"n_gpu_layers\": %d, "
                "\"n_threads\": %d, \"n_threads_batch\": %d, \"pp\": %u, \"tg\": %u, \"n_kv\": %u, "
                "\"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f }\n",
                n_kv_max, params.n_batch, params.n_ubatch, (int)params.flash_attn_type, params.n_gpu_layers,
                cparams.n_threads, cparams.n_threads_batch,
                pp, tg, n_kv, t_pp, speed_pp, t_tg, speed_tg);
        } else {
            LOG("|%6u | %6u | %6u | %8.3f | %8.2f | %8.3f | %8.2f |\n",
                pp, tg, n_kv, t_pp, speed_pp, t_tg, speed_tg);
        }

        ++i_loop;
    }

    llama_perf_context_print(ctx);
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
