# Fork Merge Map — `ik-features` @ `fe049ff54`

Generated via trial `git merge-tree` (no working-tree changes). Rollback tag `pre-fork-merge` = `fe049ff54`.

## Summary

| Fork | Branch | Shared history? | ahead / behind | Conflicting files | Verdict |
|------|--------|-----------------|----------------|-------------------|---------|
| **ik_llama** | `main` | ❌ none | n/a | n/a | **Not git-mergeable.** Hand-port / cherry-pick only (current repo workflow). |
| **prismml** | `prism` | ✅ `d104cf1` | 552 / 4 | **4** | Small. 1 semantic (quant kernels), 3 mechanical. |
| **turboquant** | `feature/turboquant-kv-cache` | ✅ `60fc495` | 182 / 204 | **13** | Medium. KV-cache + loader touch core `src/llama-*`. |
| **ggml upstream** | `master` | ✅ `d05fe1d` | 388 / 508 | **22** | Large. Mainline rewriting same ggml kernels prism customized. |

> Earlier "18/65/114" counts were inflated — they included `Auto-merging` (clean) lines. True conflict files are 4 / 13 / 22.

---

## ik_llama — UNRELATED HISTORY
`git merge` refuses (`refusing to merge unrelated histories`). No common ancestor.
Every existing `ik_llama:` commit on this branch is `retroheim`-authored hand-port. Continue that workflow: pick a specific `ik_llama/ik/<feature>` branch, cherry-pick or re-implement, build, test. `--allow-unrelated-histories` would conflict every file — not viable.

---

## prismml/prism — 4 commits behind

```
c9d528d85 ggml-cpu: AVX-512-VNNI dot-products for Q1_0/Q2_0 (#37)   <- SEMANTIC conflict
747eb3686 Merge #32 Vort3xed/vulkan-q2_0-kernel                      <- merge commit
990eb9cf1 fix: fp32_to_fp16 saturates finite overflow to inf        <- likely clean
13dec14a3 vulkan: Q2_0                                               <- vulkan conflicts
```

Conflicting files:
- `ggml/src/ggml-cpu/quants.c` — **SEMANTIC**. prismml folds 128-group logic into `ggml_vec_dot_q1_0_q8_0_generic` (`qk=QK1_0`) and has **no** separate `g128` function; HEAD keeps both `q1_0` (`QK8_0`) and `g128` as distinct kernels. Line-merge spliced incompatible bodies → compiles but wrong dot-products. Needs hand-resolution with kernel intent understood.
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp` — Q2_0 vulkan path vs HEAD turbo-quant additions.
- `ggml/src/ggml-vulkan/vulkan-shaders/copy_to_quant.comp` — large block (~130 lines) Q2_0 quantize vs HEAD turbo.
- `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` — `set_rows` type-list: union needed (`q2_0` + HEAD's `turbo2_0/3_0/4_0/tq4_1s`).

**Recommended:** cherry-pick `990eb9cf1` (fp16 fix) first — likely clean. Then hand-resolve the AVX512VNNI + vulkan-Q2_0 set as a unit (they share the Q1_0/Q2_0 theme). The `vulkan-shaders-gen` list is a trivial union.

---

## turboquant/feature/turboquant-kv-cache — 204 behind, 13 conflicts

Conflicting files:
```
README.md
common/arg.cpp                          (CLI flags)
common/speculative.cpp
convert_hf_to_gguf.py
ggml/src/ggml-cpu/arch-fallback.h
ggml/src/ggml-cuda/ggml-cuda.cu
ggml/src/ggml-metal/ggml-metal-ops.cpp
ggml/src/ggml-vulkan/ggml-vulkan.cpp
src/llama-context.cpp                   (core)
src/llama-model-loader.cpp / .h         (core loader)
src/llama-model.cpp / .h                (core model)
```
KV-cache feature touches the same core `src/llama-model*` / loader files that `ik-features` actively edits (split-mode-graph work). High overlap risk — resolve only after deciding which KV-cache representation wins. This is its own project, not a drop-in.

---

## ggml/master — 508 behind, 22 conflicts

Conflicting files concentrated in CUDA flash-attention + metal + vulkan + core loader:
```
.devops/nix/package.nix, README.md, common/arg.cpp, common/speculative.cpp,
convert_hf_to_gguf.py, ggml/src/ggml-cpu/arch-fallback.h,
ggml/src/ggml-cuda/{fattn-common.cuh,fattn-mma-f16.cuh,fattn.cu,ggml-cuda.cu},
ggml/src/ggml-metal/{ggml-metal-device.cpp,.h},
ggml/src/ggml-vulkan/ggml-vulkan.cpp,
ggml/src/ggml-vulkan/vulkan-shaders/{dequant_funcs_cm2.glsl,flash_attn_base.glsl,vulkan-shaders-gen.cpp},
src/llama-context.cpp, src/llama-kv-cache.cpp,
src/llama-model-loader.{cpp,h}, src/llama-model.{cpp,h}
```
This is mainline llama.cpp moving the same flash-attn / kv-cache / loader code prism heavily forked. 508 commits = many intermediate rewrites. Merging as one commit buries which upstream change caused each conflict. **Prefer rebasing prism's deltas onto a chosen upstream tag, or selective backport of specific upstream fixes** — not a single squashed merge.

---

## Recommended order (lowest risk → highest)

1. `prismml/prism` cherry-pick `990eb9cf1` (fp16 fix) — verify clean.
2. `prismml` AVX512VNNI + vulkan-Q2_0 — hand-resolve quant kernels, build + `test-backend-ops`.
3. `turboquant` KV-cache — **only** after deciding interaction with `ik-features` split-mode-graph KV work.
4. `ggml/master` — separate effort; selective backport, not blanket merge.

Build/test gate between every step. No step touches the tree until you approve it.
